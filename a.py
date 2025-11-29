"""
hollow_milp_builder.py
Script auxiliar para gerar um MILP (Docplex) do problema de rota com poderes, alavancas e estações.
Coloque esse código em uma célula do seu notebook ou rode como script.
Precisa de: pandas, networkx, docplex (instale com pip install docplex), optionally cplex installation.

Arquivo de entrada esperado:
- /mnt/data/pontos.csv  (colunas: name,x,y)
- /mnt/data/segmentos.csv  (colunas: name,start,end,length)

Este script:
  1) Lê os CSVs e monta o grafo físico.
  2) Carrega um arquivo de configuração JSON com nós especiais (poderes, sonhadores, estações, alavancas, start, exit, etc).
  3) Faz expansão limitada de estados (node + bitmask de poderes, sonhadores, alavancas, estações).
  4) Gera as arestas entre estados.
  5) Constrói um MILP com docplex para encontrar caminho mínimo no grafo expandido.
  6) Exporta o modelo .lp.

O arquivo de configuração JSON deve ter formato aproximado:

{
  "start": "CidadeDasLagrimas",
  "exit":  "TemploDoOvoPreto",

  "sonhadores": ["Monomon", "Lurien", "Herrah"],

  "powers": {
    "ferrao": "FerroDosSonhos_Node"
  },

  "requires": {
    "A": ["ferrao"],
    "(A,B)": ["ferrao"] 
  },

  "levers": {
    "LeverNode": [["A","B"], ["C","D"]]
  },

  "stations": ["Estacao1", "Estacao2"],
  "teleport_cost": 0.1
}

"""

import os
import json
import pickle
import pandas as pd
import networkx as nx
from itertools import product
from docplex.mp.model import Model


################################################################################
# 1. CARREGAR ARQUIVOS CSV
################################################################################

PONTOS_PATH = "/mnt/data/pontos.csv"
SEGMENTOS_PATH = "/mnt/data/segmentos.csv"
CONFIG_PATH = "/mnt/data/hollow_config.json"

if not os.path.exists(PONTOS_PATH):
    raise FileNotFoundError("Arquivo pontos.csv não encontrado em /mnt/data")

if not os.path.exists(SEGMENTOS_PATH):
    raise FileNotFoundError("Arquivo segmentos.csv não encontrado em /mnt/data")

df_p = pd.read_csv(PONTOS_PATH)
df_s = pd.read_csv(SEGMENTOS_PATH)

print(f"Carregado: {len(df_p)} pontos e {len(df_s)} segmentos.")


################################################################################
# 2. MONTA O GRAFO FÍSICO
################################################################################

G = nx.DiGraph()

# Adiciona vértices
for _, row in df_p.iterrows():
    G.add_node(row["name"], x=row["x"], y=row["y"])

# Adiciona arestas
for _, row in df_s.iterrows():
    u = row["start"]
    v = row["end"]
    w = float(row["length"])
    G.add_edge(u, v, weight=w)


################################################################################
# 3. CARREGA CONFIGURAÇÃO (SE NÃO EXISTE, CRIA TEMPLATE)
################################################################################

DEFAULT_CONFIG = {
  "start": "",
  "exit": "",
  "sonhadores": [],
  "powers": {},        # {power_id: node_name}
  "requires": {},      # {"NodeName": ["power_id"], "(u,v)": ["power_id"]}
  "levers": {},        # {"leverNode": [["u","v"], ...]}
  "stations": [],
  "teleport_cost": 1.0
}

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print("\nArquivo de configuração TEMPLATE criado em /mnt/data/hollow_config.json")
    print("Edite esse arquivo com os nomes corretos antes de rodar a expansão.")
    exit(0)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

print("\nConfiguração carregada.")


################################################################################
# 4. CRIA MAPEAMENTO DE PODERES/SONHADORES/ALAVANCAS/ESTAÇÕES
################################################################################

# Ordenar listas para definição dos bits
powers_list = sorted(config.get("powers", {}).keys())
sonhadores_list = sorted(config.get("sonhadores", []))
levers_list = sorted(config.get("levers", {}).keys())
stations_list = sorted(config.get("stations", []))

POWER_BITS = {p: i for i, p in enumerate(powers_list)}
SONHADOR_BITS = {s: i for i, s in enumerate(sonhadores_list)}
LEVER_BITS = {l: i for i, l in enumerate(levers_list)}
STATION_BITS = {s: i for i, s in enumerate(stations_list)}

NPOW = len(POWER_BITS)
NSON = len(SONHADOR_BITS)
NLEV = len(LEVER_BITS)
NSTA = len(STATION_BITS)


################################################################################
# 5. FUNÇÕES DE CHECAGEM DE PRÉ-REQUISITOS
################################################################################

def has_required(state, node, u=None, v=None):
    req = config.get("requires", {})
    needed = []

    if node in req:
        needed += req[node]
    if u is not None and v is not None:
        key = f"({u},{v})"
        if key in req:
            needed += req[key]

    for p in needed:
        if p not in POWER_BITS:
            continue
        bit = POWER_BITS[p]
        if not ((state[2] >> bit) & 1):
            return False
    return True


################################################################################
# 6. EXPANSÃO LIMITADA DE ESTADOS
################################################################################

MAX_STATES = 20000
MAX_DEPTH  = 80

start_node = config["start"]
exit_node  = config["exit"]

if start_node not in G.nodes:
    raise ValueError("Start node não existe no grafo.")
if exit_node not in G.nodes:
    raise ValueError("Exit node não existe no grafo.")

from collections import deque

start_state = (start_node, 0, 0, 0, 0)
Q = deque([(start_state, 0)])

state_to_id = {}
id_to_state = []
edges_state = []

def register_state(s):
    if s not in state_to_id:
        idx = len(id_to_state)
        state_to_id[s] = idx
        id_to_state.append(s)
    return state_to_id[s]

register_state(start_state)

print("\nIniciando expansão de estados...\n")

while Q and len(id_to_state) < MAX_STATES:
    state, depth = Q.popleft()

    if depth >= MAX_DEPTH:
        continue

    node, powMask, sonMask, levMask, staMask = state

    # coleta de poderes
    for p, nd in config.get("powers", {}).items():
        if nd == node:
            b = POWER_BITS[p]
            powMask |= (1 << b)

    # coleta de sonhadores
    if node in SONHADOR_BITS:
        b = SONHADOR_BITS[node]
        sonMask |= (1 << b)

    # alavancas
    if node in LEVER_BITS:
        b = LEVER_BITS[node]
        levMask |= (1 << b)

    # estações
    if node in STATION_BITS:
        b = STATION_BITS[node]
        staMask |= (1 << b)

    processed_state = (node, powMask, sonMask, levMask, staMask)
    sid = register_state(processed_state)

    # Transições normais
    for nxt in G.successors(node):
        if not has_required(processed_state, nxt, node, nxt):
            continue

        w = G[node][nxt]["weight"]
        new_state = (nxt, powMask, sonMask, levMask, staMask)
        nid = register_state(new_state)
        edges_state.append((sid, nid, w))

        if nid == state_to_id[new_state] and (new_state, depth+1) not in Q:
            Q.append((new_state, depth+1))

    # teleport por estações
    if staMask != 0:
        cost_tp = config.get("teleport_cost", 1.0)
        for st in stations_list:
            b = STATION_BITS[st]
            if staMask & (1 << b):
                new_state = (st, powMask, sonMask, levMask, staMask)
                nid = register_state(new_state)
                edges_state.append((sid, nid, cost_tp))

                if nid == state_to_id[new_state] and (new_state, depth+1) not in Q:
                    Q.append((new_state, depth+1))

print(f"\nExpansão finalizada:")
print(f"- {len(id_to_state)} estados")
print(f"- {len(edges_state)} arestas entre estados")


################################################################################
# 7. SALVA O GRAFO EXPANDIDO
################################################################################

with open("/mnt/data/expanded_state_graph.pkl", "wb") as f:
    pickle.dump({
        "state_to_id": state_to_id,
        "id_to_state": id_to_state,
        "edges_state": edges_state
    }, f)

print("\nGrafo de estados salvo em /mnt/data/expanded_state_graph.pkl")


################################################################################
# 8. CONSTRÓI O MILP DOCPLEX
################################################################################

mdl = Model("HK_route_minpath")

# variáveis: x[from][to][k], mas usamos único arc: x_ij
x = mdl.binary_var_dict(
    {(i, j): True for i, j, _ in edges_state},
    name="x"
)

# objetivo
mdl.minimize(mdl.sum(w * x[(i, j)] 
                     for (i, j, w) in edges_state))

# fluxo
start_ids = [i for i, st in enumerate(id_to_state) if st[0] == start_node]
exit_ids  = [i for i, st in enumerate(id_to_state)
             if st[0] == exit_node and st[2] == (1 << NSON) - 1]

# Start: 1 saída
mdl.add(mdl.sum(x[(i, j)] for (i, j, _) in edges_state if i in start_ids) == 1)

# Exits: 1 chegada
mdl.add(mdl.sum(x[(i, j)] for (i, j, _) in edges_state if j in exit_ids) == 1)

# Estados intermediários
for s in range(len(id_to_state)):
    if s in start_ids or s in exit_ids:
        continue
    in_arcs = mdl.sum(x[(i, j)] for (i, j, _) in edges_state if j == s)
    out_arcs = mdl.sum(x[(i, j)] for (i, j, _) in edges_state if i == s)
    mdl.add(in_arcs == out_arcs)

# Exportar
LP_PATH = "/mnt/data/hollow_model.lp"
mdl.export_as_lp(LP_PATH)

print(f"\nModelo MILP exportado para {LP_PATH}\n")
print("Pronto. Abra no CPLEX ou carregue via docplex para resolver.")
