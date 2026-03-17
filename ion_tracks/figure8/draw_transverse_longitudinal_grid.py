import glob
import os
import re

import MDemon as md
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

xyz_files = sorted(glob.glob('*.xyz'))
xyz_files = [f for f in xyz_files if os.path.isfile(f)]

# plt.rc('text', usetex=True)
plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

if len(xyz_files) == 0:
    raise RuntimeError('No .xyz files found in current directory')

n_cols = 5

def parse_mev_from_filename(path: str):
    name = os.path.basename(path)
    m = re.search(r'(\d+(?:\.\d+)?)\s*MeV', name, flags=re.IGNORECASE)
    if m is None:
        return None
    return float(m.group(1))


def sort_key(path: str):
    name = os.path.basename(path).lower()
    if 'pristine' in name:
        return (0, 0.0, name)
    e = parse_mev_from_filename(path)
    if e is None:
        return (0, 1.0, name)
    if abs(e - 200.0) < 1e-9:
        return (2, e, name)
    return (1, e, name)


xyz_files = sorted(xyz_files, key=sort_key)
files_to_plot = xyz_files[:n_cols]

universes = []
all_coords = []
all_ga = []
all_o = []

for path in files_to_plot:
    u = md.Universe(path)
    coords = u.atoms.coordinate
    center_of_mass = np.mean(coords, axis=0)
    coords = coords - center_of_mass

    a1 = u.atoms[0]
    e1 = a1.element
    m1 = a1.mass

    if m1 > 20:
        element_dic = {a1.species: "Ga", 1 if a1.species == 2 else 2: "O"}
    else:
        element_dic = {a1.species: "O", 1 if a1.species == 2 else 2: "Ga"}
    elements = np.array([element_dic[i] for i in u.atoms.species])
    ga_coords = coords[elements == 'Ga']
    o_coords = coords[elements == 'O']

    universes.append(u)
    all_coords.append(coords)
    all_ga.append(ga_coords)
    all_o.append(o_coords)

coords_stack = np.concatenate(all_coords, axis=0)
x_lim = 50
y_lim = 50
z_lim = 50

pad = 1.05
x_lim *= pad
y_lim *= pad
z_lim *= pad

figsize = 22
fontsize = 10

zorder_ga = 3
zorder_o = 2

titlelist = [r"$\boldsymbol{S_e} = 15.68$ keV/nm",
            r"$S_e = 19.18$ keV/nm",
            r"$S_e = 22.16$ keV/nm",
            r"$S_e = 26.05$ keV/nm",
            r"$S_e = 36.24$ keV/nm"]

n = 100
x0 = 10 * n
y0 = 10 * n
dx = int(x0 / 10)
dy = int(y0 / 5)
M = int(n_cols * x0 + (n_cols - 1) * dx)
N = int(2 * y0 + dy)

fig = plt.figure(figsize=(figsize, N / (M / figsize)))
gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))

color_ga = '#0d47a1'
color_o = '#d84315'
marker_size = 1.4
alpha = 0.8

for col_idx in range(n_cols):
    x_start = col_idx * (x0 + dx)
    x_end = x_start + x0

    ax_t = fig.add_subplot(gs[0:y0, x_start:x_end])
    ax_l = fig.add_subplot(gs[y0 + dy:y0 + dy + y0, x_start:x_end])

    if col_idx >= len(files_to_plot):
        ax_t.axis('off')
        ax_l.axis('off')
        continue

    ga_coords = all_ga[col_idx]
    o_coords = all_o[col_idx]

    ax_t.scatter(ga_coords[:, 0], ga_coords[:, 2], s=marker_size, alpha=alpha, color=color_ga, edgecolor='none', label='Ga',zorder=zorder_ga)
    ax_t.scatter(o_coords[:, 0], o_coords[:, 2], s=marker_size * 0.6, alpha=alpha, color=color_o, edgecolor='none', label='O',zorder=zorder_o)
    ax_t.set_xlabel('X (Å)', fontsize=fontsize, fontweight='bold')
    if col_idx == 0:
        ax_t.set_ylabel('Z (Å)', fontsize=fontsize, fontweight='bold')
    else:
        ax_t.set_ylabel('')
    ax_t.tick_params(labelsize=fontsize - 1)
    ax_t.tick_params(axis='y', rotation=90)
    for label in ax_t.get_yticklabels():
        label.set_verticalalignment('center')
    ax_t.set_aspect('equal', adjustable='box')
    ax_t.set_xlim(-x_lim, x_lim)
    ax_t.set_ylim(-z_lim, z_lim)
    ax_t.text(
        0.5,
        1.02,
        titlelist[col_idx],
        transform=ax_t.transAxes,
        fontsize=fontsize + 4,
        # fontweight='bold',
        ha='center',
        va='bottom'
    )

    ax_l.scatter(ga_coords[:, 0], ga_coords[:, 1], s=marker_size, alpha=alpha, color=color_ga, edgecolor='none',zorder=zorder_ga)
    ax_l.scatter(o_coords[:, 0], o_coords[:, 1], s=marker_size * 0.6, alpha=alpha, color=color_o, edgecolor='none',zorder=zorder_o)
    ax_l.set_xlabel('X (Å)', fontsize=fontsize, fontweight='bold')
    if col_idx == 0:
        ax_l.set_ylabel('Y (Å)', fontsize=fontsize, fontweight='bold')
    else:
        ax_l.set_ylabel('')
    ax_l.tick_params(labelsize=fontsize - 1)
    ax_l.tick_params(axis='y', rotation=90)
    for label in ax_l.get_yticklabels():
        label.set_verticalalignment('center')
    ax_l.set_aspect('equal', adjustable='box')
    ax_l.set_xlim(-x_lim, x_lim)
    ax_l.set_ylim(-y_lim, y_lim)
    ax_t.legend(loc='upper right', fontsize=fontsize-1, markerscale=5)
plt.tight_layout()
plt.savefig('figure8_transverse_longitudinal_grid.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'figure8_transverse_longitudinal_grid.png'")
