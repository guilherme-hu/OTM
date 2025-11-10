# -*- coding: utf-8 -*-
"""
Visualiza `pontos.csv` e `segmentos.csv` gerando `mapa_plot.png`.
- Lê `pontos.csv` (name,x,y)
- Lê `segmentos.csv` (name,start,end,length)
- Desenha pontos com rótulos e segmentos como linhas
- Salva imagem em `mapa_plot.png` no mesmo diretório

Uso:
    python visualize_mapa.py

Se quiser visualização interativa, abra o arquivo PNG ou modifique para exibir com plt.show().
"""
import csv
from pathlib import Path
import math
import re

# for non-interactive environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.lines import Line2D

BASE = Path(r"c:\Users\guilh\OneDrive\Faculdade\FACULDADE\6° período\OTM\Trab")
PONTOS_CSV = BASE / 'pontos.csv'
SEGMENTOS_CSV = BASE / 'segmentos.csv'
OUT_PNG = BASE / 'mapa_plot.png'

if not PONTOS_CSV.exists() or not SEGMENTOS_CSV.exists():
    print('Arquivos `pontos.csv` ou `segmentos.csv` não encontrados no diretório:', BASE)
    raise SystemExit(1)

# lê pontos
points = {}
with PONTOS_CSV.open('r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get('name','').strip()
        x = row.get('x','').strip()
        y = row.get('y','').strip()
        try:
            xf = float(x)
            yf = float(y)
            points[name] = (xf, yf)
        except Exception:
            # ignora pontos sem coords válidas
            continue

# lê segmentos
segments = []
missing_points = set()
with SEGMENTOS_CSV.open('r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get('name','').strip()
        start = row.get('start','').strip()
        end = row.get('end','').strip()
        length = row.get('length','').strip()
        # store
        segments.append({'name': name, 'start': start, 'end': end, 'length': length})
        if start not in points:
            missing_points.add(start)
        if end not in points:
            missing_points.add(end)

# Prepare plot
xs = [c[0] for c in points.values()]
ys = [c[1] for c in points.values()]

if xs and ys:
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
else:
    minx = miny = 0
    maxx = maxy = 1

# margin
dx = maxx - minx if maxx > minx else 1.0
dy = maxy - miny if maxy > miny else 1.0
margin = 0.05
padx = dx * margin
pady = dy * margin

fig_w = 10
# adapt figure size to aspect and to number of points (wider when many points)
num_points = max(1, len(points))
density_scale = min(3.0, 1.0 + (num_points / 300.0))
fig_h = max(6, (dy/dx) * fig_w) if dx > 0 else fig_w
fig = plt.figure(figsize=(fig_w * density_scale, fig_h * density_scale))
ax = fig.add_subplot(1,1,1)

# Draw segments first (so points/labels are on top)
for seg in segments:
    s = seg['start']
    e = seg['end']
    if s in points and e in points:
        x1,y1 = points[s]
        x2,y2 = points[e]
        ax.plot([x1,x2],[y1,y2], color='gray', linewidth=1, zorder=1)
    else:
        # skip drawing if missing
        continue

# Draw points
px = []
py = []
plabels = []
pcolors = []
def point_color(name: str) -> str:
    n = (name or '').strip()
    nl = n.lower()
    # Stations / trains -> blue (explicit)
    if 'estacao' in nl or 'trem' in nl:
        return 'tab:blue'
    # Sonhadores (objetivos) -> gold (very special)
    if re.search(r'(?i)sonh', n):
        return 'gold'
    # Alavancas (contains 'Alavanca') -> purple
    if re.search(r'(?i)alavanc', n):
        return 'tab:purple'
    # Nodes that require a power: start with literal N followed by letters (e.g. NLL, NL, NVR)
    if re.match(r'^N[A-Z]+[0-9]*$', n):
        return 'tab:orange'
    # Simple nodes: letters followed by digits (A1, H11, K8)
    if re.match(r'^[A-Za-z]+[0-9]+$', n):
        return 'tab:gray'
    # Big named things (single word, mostly letters, length >=5) -> highlight
    if len(n) >= 5 and re.match(r'^[A-Za-z]+$', n):
        return 'tab:green'
    # fallback: other special names
    return 'tab:red'
for name,(x,y) in points.items():
    px.append(x)
    py.append(y)
    plabels.append(name)
    pcolors.append(point_color(name))
# Compute a sensible marker size based on map scale and point density
try:
    # estimate closest neighbor distance (O(n^2) but n is moderate here)
    min_dist = None
    coords = list(points.values())
    n = len(coords)
    for i in range(n):
        x1,y1 = coords[i]
        for j in range(i+1, n):
            x2,y2 = coords[j]
            d = math.hypot(x2-x1, y2-y1)
            if d == 0:
                continue
            if min_dist is None or d < min_dist:
                min_dist = d
    if not min_dist or min_dist <= 0:
        min_dist = max(dx, dy) * 0.005
except Exception:
    min_dist = max(dx, dy) * 0.005

# marker size in points^2 (approx); inversely proportional to density
marker_size = max(6, min(60, (min(dx,dy) / (min_dist + 1e-9)) * 3))
ax.scatter(px, py, c=pcolors, s=marker_size, edgecolor='k', linewidth=0.3, zorder=2)

# Annotate all labels but scale font size and offset according to local density
x_range = maxx - minx if maxx>minx else 1.0
y_range = maxy - miny if maxy>miny else 1.0

# base offset: fraction of min distance between points
base_off = min_dist * 0.25
# ensure a minimum offset in data units
min_off = max(x_range, y_range) * 0.002
off = max(base_off, min_off)

# font sizing: smaller fonts when many points close together
font_size = max(4, min(10, int(8 * (min_dist / (max(dx, dy) * 0.05) + 0.2))))

# Place labels with a simple collision-avoidance heuristic:
# try a series of offsets and pick the first that doesn't collide with
# previously placed labels (measured by distance in data coordinates).
placed_labels = []  # list of (lx,ly) label anchor positions
label_sep = max(min_dist * 0.6, max(x_range, y_range) * 0.01)
offset_options = [
    (off, off), (off, -off), (-off, off), (-off, -off),
    (2*off, 0), (0, 2*off), (-2*off, 0), (0, -2*off)
]

for name, (x, y) in points.items():
    placed = False
    for dx_off, dy_off in offset_options:
        lx = x + dx_off
        ly = y + dy_off
        collision = False
        for px_l, py_l in placed_labels:
            if math.hypot(lx - px_l, ly - py_l) < label_sep:
                collision = True
                break
        if not collision:
            ax.annotate(name, (lx, ly), fontsize=font_size,
                        bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.6, edgecolor='none'),
                        zorder=3)
            placed_labels.append((lx, ly))
            placed = True
            break
        if not placed:
            # as a last resort, place with very small font and lower opacity to avoid clutter
            ax.annotate(name, (x + off, y + off), fontsize=max(3, font_size - 2),
                        color='gray', alpha=0.6,
                        bbox=dict(boxstyle='round,pad=0.08', facecolor='white', alpha=0.6, edgecolor='none'),
                        zorder=2)

ax.set_xlim(minx - padx, maxx + padx)
ax.set_ylim(miny - pady, maxy + pady)
ax.set_aspect('equal', adjustable='box')
ax.set_title('Mapa — pontos e segmentos')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Legend with counts
ax.text(0.99, 0.01, f'Pontos: {len(points)}\nSegmentos: {len(segments)}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# color legend for point types
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='Estacao/Trem', markerfacecolor='tab:blue', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Sonhadores (objetivos)', markerfacecolor='gold', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Alavanca', markerfacecolor='tab:purple', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Node N[A-Z]+', markerfacecolor='tab:orange', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Simple (A1,H11)', markerfacecolor='tab:gray', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Large name (Frase)', markerfacecolor='tab:green', markersize=6),
    Line2D([0], [0], marker='o', color='w', label='Outros', markerfacecolor='tab:red', markersize=6),
]
ax.legend(handles=legend_handles, loc='upper left', fontsize=8, framealpha=0.9)

# Save
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200)
print('Imagem salva em:', OUT_PNG)
print('Pontos lidos:', len(points))
print('Segmentos lidos:', len(segments))
if any(missing_points):
    # remove empty names
    missing = sorted([m for m in missing_points if m])
    print('Aviso: os seguintes nomes referenciados em segmentos não foram encontrados em pontos.csv:')
    for m in missing[:50]:
        print(' -', m)
    if len(missing) > 50:
        print(' ... e mais', len(missing)-50, 'itens')
else:
    print('Todos os pontos referenciados em segmentos foram encontrados.')

print('Pronto.')

# --- Second plot: overlay points on background image (image.png) ---
IMG_PATH = BASE / 'image.png'
OUT_PNG_BG = BASE / 'mapa_plot_bg.png'

if IMG_PATH.exists():
    try:
        img = mpimg.imread(str(IMG_PATH))
        # flip the image vertically so it displays right-side-up on the plot
        try:
            img = np.flipud(img)
        except Exception:
            # if flip fails (e.g., unexpected array shape), fall back to original
            pass
        ih, iw = img.shape[0], img.shape[1]
        # map image width to x in [0,700]; scale y accordingly
        scale = 700.0 / float(iw)
        img_height_scaled = ih * scale

        fig2_w = max(10, fig_w * density_scale)
        fig2_h = max(6, (img_height_scaled / 700.0) * fig2_w)
        fig2 = plt.figure(figsize=(fig2_w, fig2_h))
        ax2 = fig2.add_subplot(1,1,1)

        # show image with origin at lower so (0,0) is bottom-left
        ax2.imshow(img, extent=[0, 700, 0, img_height_scaled], origin='lower')

        # draw segments on top (plot using image coords: bottom-left = (0,0))
        for seg in segments:
            s = seg['start']
            e = seg['end']
            if s in points and e in points:
                x1,y1 = points[s]
                x2,y2 = points[e]
                ax2.plot([x1,x2],[y1,y2], color='gray', linewidth=1, zorder=2)

        # draw points
        # use point coordinates directly (image origin is set to lower so (0,0) is bottom-left)
        px2 = [c[0] for c in points.values()]
        py2 = [c[1] for c in points.values()]
        # adapt marker size for image plot
        marker_size2 = marker_size
        # build color list for image overlay in same order as points.values()
        pkeys = list(points.keys())
        colors2 = [point_color(k) for k in pkeys]
        ax2.scatter(px2, py2, c=colors2, s=marker_size2, edgecolor='k', linewidth=0.3, zorder=3)

        # place labels using same heuristic but constrained to image extent
        placed_labels = []
        label_sep = max(min_dist * 0.6, max(700.0, img_height_scaled) * 0.01)
        offset_options = [
            (off, off), (off, -off), (-off, off), (-off, -off),
            (2*off, 0), (0, 2*off), (-2*off, 0), (0, -2*off)
        ]
        for name, (x, y) in points.items():
            placed = False
            for dx_off, dy_off in offset_options:
                # compute candidate label position in image coords
                lx = x + dx_off
                ly = y + dy_off
                # ensure label anchor inside image bounds
                if lx < 0 or lx > 700 or ly < 0 or ly > img_height_scaled:
                    continue
                collision = False
                for px_l, py_l in placed_labels:
                    if math.hypot(lx - px_l, ly - py_l) < label_sep:
                        collision = True
                        break
                if not collision:
                    ax2.annotate(name, (lx, ly), fontsize=font_size,
                                     bbox=dict(boxstyle='round,pad=0.10', facecolor='white', alpha=0.7, edgecolor='none'),
                                     zorder=4)
                    placed_labels.append((lx, ly))
                    placed = True
                    break
            if not placed:
                ax2.annotate(name, (x + off, y + off), fontsize=max(3, font_size - 2),
                             color='gray', alpha=0.6,
                             bbox=dict(boxstyle='round,pad=0.08', facecolor='white', alpha=0.6, edgecolor='none'),
                             zorder=3)

        ax2.set_xlim(0, 700)
        ax2.set_ylim(0, img_height_scaled)
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_title('Mapa sobre imagem — pontos e segmentos')

        fig2.tight_layout()
        fig2.savefig(OUT_PNG_BG, dpi=200)
        print('Imagem com fundo salva em:', OUT_PNG_BG)
    except Exception as exc:
        print('Falha ao gerar imagem com fundo:', exc)
else:
    print('Arquivo de imagem `image.png` não encontrado em:', IMG_PATH)
