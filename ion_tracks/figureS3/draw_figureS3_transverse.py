import glob
import os
import re
import pickle
from pathlib import Path

import MDemon as md
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Set path to raw_data directory
raw_data_dir = Path(__file__).parent / 'raw_data'
xyz_files = sorted(glob.glob(str(raw_data_dir / '*.xyz')))

plt.rcParams['font.family'] = 'Arial'

if len(xyz_files) == 0:
    raise RuntimeError('No .xyz files found in raw_data directory')

# Define 12 ion irradiation conditions
conditions = [
    ('Ta', 20, 11.29, [0,0,0]),
    ('Ta', 30, 15.68, [0,0,0]),
    ('Ta', 40, 19.18, [26,23,24]),
    ('Ta', 50, 22.16, [42,38,42]),
    ('Ta', 70, 26.05, [59,60,58]),
    ('Ta', 100, 30.08, [71,73,69]),
    ('Ta', 200, 36.24, [80,79,82]),
    ('Ta', 1000, 39.26, [75,73,72]),
    ('Au', 1000, 42.92, [83,81,80]),
]
def parse_filename(path: str):
    """Parse ion type, energy, and iteration from filename"""
    name = os.path.basename(path)
    m = re.match(r'([A-Za-z]+)_(\d+)MeV_iter(\d+)\.xyz', name)
    if m:
        ion = m.group(1)
        energy = int(m.group(2))
        iteration = int(m.group(3))
        return ion, energy, iteration
    return None, None, None

# Group files by condition and iteration
file_dict = {}
for path in xyz_files:
    ion, energy, iteration = parse_filename(path)
    if ion and energy and iteration:
        key = (ion, energy)
        if key not in file_dict:
            file_dict[key] = {}
        file_dict[key][iteration] = path

# Parameters from reference script - use fixed limits (needed for data processing cutoff)
x_lim = 50
y_lim = 50
z_lim = 50
pad = 1.05
x_lim *= pad
y_lim *= pad
z_lim *= pad

# Cache file path
cache_file = Path(__file__).parent / 'processed_data_cache.pkl'

# Try to load cached data
if cache_file.exists():
    print(f"Loading cached data from {cache_file.name}...")
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
        all_ga = cache_data['all_ga']
        all_o = cache_data['all_o']
    print("Cache loaded successfully!")
else:
    print("Cache not found. Processing raw data...")
    # Load all data
    all_ga = []
    all_o = []

    for condition in conditions:
        ion, energy, se, d_list = condition
        key = (ion, energy)
        row_data_ga = []
        row_data_o = []
        
        for iter_num in [1, 2, 3]:
            if key in file_dict and iter_num in file_dict[key]:
                path = file_dict[key][iter_num]
                print(f"  Processing {os.path.basename(path)}...")
                u = md.Universe(path)
                coords = u.atoms.coordinate
                center_of_mass = np.mean(coords, axis=0)
                coords = coords - center_of_mass
                
                a1 = u.atoms[0]
                m1 = a1.mass
                
                if m1 > 20:
                    element_dic = {a1.species: "Ga", 1 if a1.species == 2 else 2: "O"}
                else:
                    element_dic = {a1.species: "O", 1 if a1.species == 2 else 2: "Ga"}
                elements = np.array([element_dic[i] for i in u.atoms.species])
                ga_coords = coords[elements == 'Ga']
                o_coords = coords[elements == 'O']
                
                # Apply cutoff: only keep atoms within [-x_lim, x_lim] x [-z_lim, z_lim] for xz plane view
                ga_mask = (np.abs(ga_coords[:, 0]) <= x_lim) & (np.abs(ga_coords[:, 2]) <= z_lim)
                o_mask = (np.abs(o_coords[:, 0]) <= x_lim) & (np.abs(o_coords[:, 2]) <= z_lim)
                ga_coords = ga_coords[ga_mask]
                o_coords = o_coords[o_mask]
                
                row_data_ga.append(ga_coords)
                row_data_o.append(o_coords)
            else:
                row_data_ga.append(None)
                row_data_o.append(None)
        
        all_ga.append(row_data_ga)
        all_o.append(row_data_o)
    
    # Save to cache
    print(f"Saving processed data to cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump({'all_ga': all_ga, 'all_o': all_o}, f)
    print(f"Cache saved to {cache_file.name}")

# Parameters from reference script
n_cols = 3
figsize = 12
fontsize = 10

zorder_ga = 3
zorder_o = 2

color_ga = '#0d47a1'
color_o = '#d84315'
marker_size = 2
alpha = 0.8

# Grid setup (similar to reference script)
n = 100
x0 = 10 * n
y0 = 10 * n
dx = int(x0 / 4)
dy = int(y0 / 3)
M = int(n_cols * x0 + (n_cols - 1) * dx)

from matplotlib.patches import Circle

def plot_figure(row_start, row_end, output_filename):
    """Plot a figure with rows from row_start to row_end (inclusive)"""
    n_rows = row_end - row_start + 1
    N = int(n_rows * y0 + (n_rows - 1) * dy)
    
    fig = plt.figure(figsize=(figsize, N / (M / figsize)))
    gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))
    
    # Plot each condition and iteration
    for local_row_idx in range(n_rows):
        row_idx = row_start + local_row_idx
        ion, energy, se, d_list = conditions[row_idx]
        
        for col_idx in range(n_cols):
            iter_num = col_idx + 1
            
            y_start = local_row_idx * (y0 + dy)
            y_end = y_start + y0
            x_start = col_idx * (x0 + dx)
            x_end = x_start + x0
            
            ax = fig.add_subplot(gs[y_start:y_end, x_start:x_end])
            
            ga_coords = all_ga[row_idx][col_idx]
            o_coords = all_o[row_idx][col_idx]
            
            if ga_coords is not None and o_coords is not None:
                # Transverse cross-sectional view (xz plane)
                ax.scatter(ga_coords[:, 0], ga_coords[:, 2], s=marker_size, alpha=alpha, color=color_ga, edgecolor='none', label='Ga', zorder=zorder_ga)
                ax.scatter(o_coords[:, 0], o_coords[:, 2], s=marker_size * 0.6, alpha=alpha, color=color_o, edgecolor='none', label='O', zorder=zorder_o)
            else:
                ax.axis('off')
                continue
            
            # 画一个圈
            if d_list[col_idx] > 0:
                ax.add_patch(Circle((0, 0), d_list[col_idx]/2, fill=False, edgecolor='black', linewidth=1.5, linestyle='--', zorder=5))
              # 标注直径
                r_i = d_list[col_idx]/2+10
                angle = 54 * np.pi/180
                ax.text(-r_i*np.cos(angle), r_i*np.sin(angle), 'D = ' + str(d_list[col_idx]/10) + ' nm', 
                        backgroundcolor='white',
                        fontsize=fontsize, fontweight='bold', ha='center', va='center', color='black', zorder=6)
            else:
                pass
            
            # Labels
            ax.set_xlabel('X (Å)', fontsize=fontsize, fontweight='bold')
            ax.set_ylabel('Z (Å)', fontsize=fontsize, fontweight='bold')
            
            ax.tick_params(labelsize=fontsize - 1)
            ax.tick_params(axis='y', rotation=90)
            for label in ax.get_yticklabels():
                label.set_verticalalignment('center')
            
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-x_lim, x_lim)
            ax.set_ylim(-z_lim, z_lim)
            
            # Title (condition label)
            title = fr'{ion} {energy} MeV ($S_e={se}$ keV/nm), Iteration {iter_num}'
            ax.text(
                0.5,
                1.02,
                title,
                transform=ax.transAxes,
                fontsize=fontsize + 1,
                # fontweight='bold',
                ha='center',
                va='bottom'
                )
            
            # Legend (same as reference script)
            ax.legend(loc='upper right', fontsize=fontsize - 1, markerscale=5)
    
    plt.tight_layout()
    output_file = Path(__file__).parent / output_filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved as '{output_file}'")

# Plot first figure (first 5 rows: 0-4)
plot_figure(0, 4, 'figureS3_transverse_part1.png')

# Plot second figure (last 4 rows: 5-8)
plot_figure(5, 8, 'figureS3_transverse_part2.png')
