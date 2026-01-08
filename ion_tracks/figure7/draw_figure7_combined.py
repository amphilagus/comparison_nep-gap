import MDemon as md
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

u = md.Universe('cropped.xyz')
coordinates = u.atoms.coordinate
center_of_mass = np.mean(coordinates, axis=0)
coordinates = coordinates - center_of_mass

element_dic = {1: "Ga", 2: "O"}
elements = [element_dic[i] for i in u.atoms.species]
ga_coords = coordinates[np.array(elements) == 'Ga']
o_coords = coordinates[np.array(elements) == 'O']

rdf_ga_ga = pd.read_csv('rdf_Ga_Ga.csv')
rdf_ga_o = pd.read_csv('rdf_Ga_O.csv')
rdf_o_o = pd.read_csv('rdf_O_O.csv')

summary = pd.read_csv('distance_distributions_summary.csv')
zones = {
    'Zone 0-25 Å': pd.read_csv('distance_distributions_Zone_0-25_A.csv'),
    'Zone 25-35 Å': pd.read_csv('distance_distributions_Zone_25-35_A.csv'),
    'Zone 35-45 Å': pd.read_csv('distance_distributions_Zone_35-45_A.csv'),
    'Zone 45-60 Å': pd.read_csv('distance_distributions_Zone_45-60_A.csv')
}
fcc_stats = summary[summary['site_type'] == 'FCC']
fcc_xmin = fcc_stats['min'].min()
fcc_xmax = fcc_stats['max'].max()
fcc_bins = np.linspace(fcc_xmin, fcc_xmax, 51)

figsize = 24
fontsize = 10
plt.rcParams['font.family'] = 'Arial'

n = 100

left_x0 = 22 * n
left_x1 = int(0.6 * left_x0 * 10 / 10.5)
left_dx = int(left_x0 / 10)
left_pad_x = int(left_dx/5)
left_M = int(left_x0 + left_x1 + left_dx + left_pad_x)

right_x0 = 10 * n
right_y0 = 8 * n
right_dx = int(right_x0 / 4)
right_pad_x = int(right_dx/5)
right_dy = int(right_y0 / 7)
right_pad_y = int(right_dy/5)
right_M = int(3 * right_x0 + 2 * right_dx + right_pad_x)
right_N = int(4 * right_y0 + 3 * right_dy + 2*right_pad_y)

gap_x = int(right_dx*1.5)
M = left_M + gap_x + right_M
N = right_N

fig = plt.figure(figsize=(figsize, N / (M / figsize)))
gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))

x_left_start = left_pad_x
x_left_end = left_M
x_right_start = left_M + gap_x

gap_y = right_dy*3
top_h = left_x0
top_pad_y = right_pad_y
bottom_start = top_h + gap_y + top_pad_y

ax1 = fig.add_subplot(gs[top_pad_y:top_h + top_pad_y, x_left_start:left_x0+x_left_start])
ax2 = fig.add_subplot(gs[top_pad_y:top_h + top_pad_y, left_x0 + x_left_start + left_dx:x_left_end])

color_ga = '#0d47a1'
color_o = '#d84315'
marker_size = 1.0
alpha = 0.8

ax1.scatter(ga_coords[:, 0], ga_coords[:, 2], s=marker_size, alpha=alpha, color=color_ga, label='Ga', edgecolor='none')
ax1.scatter(o_coords[:, 0], o_coords[:, 2], s=marker_size * 0.6, alpha=alpha, color=color_o, label='O', edgecolor='none')
ax1.set_xlabel('X (Å)', fontsize=fontsize, fontweight='bold')
ax1.set_ylabel('Z (Å)', fontsize=fontsize, fontweight='bold')
ax1.legend(loc='upper right', fontsize=fontsize + 1, markerscale=5)
ax1.tick_params(labelsize=fontsize - 1)
ax1.tick_params(axis='y', rotation=90)
for label in ax1.get_yticklabels():
    label.set_verticalalignment('center')
ax1.set_aspect('equal', adjustable='box')
ax1.text(-0.09, 1.03, '(a)', transform=ax1.transAxes, fontsize=fontsize + 3, fontweight='bold', ha='left', va='top')

radii = [25, 35, 45]
for radius in radii:
    circle = Circle((0, 0), radius, fill=False, edgecolor='black', linewidth=1.5, linestyle='--', zorder=5)
    ax1.add_patch(circle)

r_i = (25) * 3/4
angle_i = np.pi / 4
ax1.text(r_i * np.cos(angle_i), r_i * np.sin(angle_i), 'I', fontsize=fontsize + 4, fontweight='bold', ha='center', va='center', color='black', zorder=6)
angle_ii = np.pi / 4
r_ii = (25 + 35) / 2
ax1.text(r_ii * np.cos(angle_ii), r_ii * np.sin(angle_ii), 'II', fontsize=fontsize + 4, fontweight='bold', ha='center', va='center', color='black', zorder=6)
angle_iii = np.pi / 4
r_iii = (35 + 45) / 2
ax1.text(r_iii * np.cos(angle_iii), r_iii * np.sin(angle_iii), 'III', fontsize=fontsize + 4, fontweight='bold', ha='center', va='center', color='black', zorder=6)
angle_iv = np.pi / 4
r_iv = 50
ax1.text(r_iv * np.cos(angle_iv), r_iv * np.sin(angle_iv), 'IV', fontsize=fontsize + 4, fontweight='bold', ha='center', va='center', color='black', zorder=6)

ax1.set_xlim(-60, 60)
ax1.set_ylim(-60, 60)

ax2.scatter(ga_coords[:, 0], ga_coords[:, 1], s=marker_size, alpha=alpha, color=color_ga, label='Ga', edgecolor='none')
ax2.scatter(o_coords[:, 0], o_coords[:, 1], s=marker_size * 0.6, alpha=alpha, color=color_o, label='O', edgecolor='none')
ax2.set_xlabel('X (Å)', fontsize=fontsize, fontweight='bold')
ax2.set_ylabel('Y (Å)', fontsize=fontsize, fontweight='bold')
ax2.tick_params(labelsize=fontsize - 1)
ax2.tick_params(axis='y', rotation=90)
for label in ax2.get_yticklabels():
    label.set_verticalalignment('center')
ax2.set_aspect('equal', adjustable='box')
ax2.text(-0.09, 1.03, '(b)', transform=ax2.transAxes, fontsize=fontsize + 3, fontweight='bold', ha='left', va='top')
ax2.set_xlim(-60, 60)
ax2.set_ylim(-105, 105)

bottom_h = N - bottom_start
bottom_pad_y = right_pad_y

rdf_dx = int((left_M - x_left_start)/ 15)
rdf_x0 = int((left_M - 2 * rdf_dx) / 3)
rdf_y_start = bottom_start
rdf_y_end = N - bottom_pad_y
ax_rdf_1 = fig.add_subplot(gs[rdf_y_start:rdf_y_end, 0:rdf_x0])

print([rdf_y_start,rdf_y_end, rdf_x0 + rdf_dx,rdf_x0 + rdf_dx + rdf_x0])
ax_rdf_2 = fig.add_subplot(gs[rdf_y_start:rdf_y_end, rdf_x0 + rdf_dx:rdf_x0 + rdf_dx + rdf_x0])
ax_rdf_3 = fig.add_subplot(gs[rdf_y_start:rdf_y_end, 2 * (rdf_x0 + rdf_dx):2 * (rdf_x0 + rdf_dx) + rdf_x0])

colors = {
    'pristine': '#1565c0',
    'core': '#d32f2f',
    'shell': '#f57c00',
    'outer': '#00897b',
    'far': '#7b1fa2'
}
linestyles = {
    'pristine': '-',
    'core': '-',
    'shell': '--',
    'outer': '--',
    'far': ':'
}
linewidths = {
    'pristine': 2.0,
    'core': 1.8,
    'shell': 1.8,
    'outer': 1.8,
    'far': 1.8
}

datasets = [
    (rdf_ga_ga, 'Ga-Ga RDF', ax_rdf_1, '(c)'),
    (rdf_ga_o, 'Ga-O RDF', ax_rdf_2, '(d)'),
    (rdf_o_o, 'O-O RDF', ax_rdf_3, '(e)')
]

for data, title, ax, label in datasets:
    r = data['r_angstrom']
    ax.plot(r, data['pristine_rdf'], color=colors['pristine'], linestyle=linestyles['pristine'], linewidth=linewidths['pristine'], label='Pristine')
    ax.plot(r, data['Irr._Core_0-15A_rdf'], color=colors['core'], linestyle=linestyles['core'], linewidth=linewidths['core'], label=r'Region I')
    ax.plot(r, data['Irr._Shell_15-30A_rdf'], color=colors['shell'], linestyle=linestyles['shell'], linewidth=linewidths['shell'], label=r'Region II')
    ax.plot(r, data['Irr._Outer_30-45A_rdf'], color=colors['outer'], linestyle=linestyles['outer'], linewidth=linewidths['outer'], label=r'Region III')
    ax.plot(r, data['Irr._Far_45-60A_rdf'], color=colors['far'], linestyle=linestyles['far'], linewidth=linewidths['far'], label=r'Region IV')
    ax.set_xlabel(r'$\boldsymbol{r}$ (Å)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(r'$\boldsymbol{g(r)}$', fontsize=fontsize, fontweight='bold')
    ax.tick_params(labelsize=fontsize - 1)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.text(-0.1, 1.08, label, transform=ax.transAxes, fontsize=fontsize + 4, fontweight='bold', ha='left', va='top')
    ax.set_ylim(bottom=0)
    ax.set_xlim(1.0, 5.5)

ax_rdf_1.legend(loc='upper left', fontsize=fontsize - 3, framealpha=0.9)

dist_colors = {
    'FCC': '#5B8DBE',
    '4-coord': '#6EAA5E',
    '6-coord': '#E6955C'
}
columns = [
    ('FCC', 'O Atoms to FCC Sites', 'O'),
    ('4-coord', 'Ga Atoms to 4-Coordinated Sites', 'Ga'),
    ('6-coord', 'Ga Atoms to 6-Coordinated Sites', 'Ga')
]
zone_names = ['Zone 0-25 Å', 'Zone 25-35 Å', 'Zone 35-45 Å', 'Zone 45-60 Å']
real_names = ['Region I', 'Region II', 'Region III', 'Region IV']

label_offset = 5

for row_idx, zone_name in enumerate(zone_names):
    zone_data = zones[zone_name]
    real_name = real_names[row_idx]

    for col_idx, (site_type, title, atom_type) in enumerate(columns):
        ax = fig.add_subplot(
            gs[
                right_pad_y + row_idx * (right_y0 + right_dy):right_pad_y + row_idx * (right_y0 + right_dy) + right_y0,
                x_right_start + col_idx * (right_x0 + right_dx):x_right_start + col_idx * (right_x0 + right_dx) + right_x0
            ]
        )

        data = zone_data[zone_data['site_type'] == site_type]['distance_angstrom'].values
        stats = summary[(summary['zone'] == zone_name) & (summary['site_type'] == site_type)]

        if len(stats) > 0 and len(data) > 0:
            mean = stats['mean'].values[0]
            ratio = stats['ratio'].values[0]

            weights = np.ones_like(data, dtype=float) / len(data)
            bins = fcc_bins if site_type == 'FCC' else 50
            ax.hist(data, bins=bins, weights=weights, color=dist_colors[site_type], alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f} Å')

            textstr = f'Mean = {mean:.4f} Å'
            if atom_type == 'Ga':
                cutoff = 0.7550
                within_pct = ratio * 100
                ax.axvline(cutoff, color='green', linestyle=':', linewidth=2)
                textstr += f'\nWithin cutoff: {within_pct:.1f}%'

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(
                0.95 if col_idx == 0 else 0.05,
                0.95,
                textstr,
                transform=ax.transAxes,
                fontsize=fontsize - 1,
                verticalalignment='top',
                horizontalalignment='right' if col_idx == 0 else 'left',
                bbox=props
            )

        if row_idx == 3:
            ax.set_xlabel(
                'Distance to Nearest ' + ('FCC Site' if site_type == 'FCC' else f'{site_type.split("-")[0]}-Coordinate Site') + ' (Å)',
                fontsize=fontsize
            )
        ax.set_ylabel('Frequency', fontsize=fontsize)

        if row_idx == 0:
            ax.set_title(title, fontsize=fontsize + 2, fontweight='bold', pad=8)

        if col_idx == 0:
            ax.text(
                -0.25,
                0.5,
                real_name,
                transform=ax.transAxes,
                fontsize=fontsize + 1,
                fontweight='bold',
                verticalalignment='center',
                horizontalalignment='center',
                rotation=90
            )

        label_idx = row_idx * len(columns) + col_idx
        label = f'({chr(97 + label_offset + label_idx)})'
        ax.text(-0.15, 1.09, label, transform=ax.transAxes, fontsize=fontsize + 4, fontweight='bold', ha='left', va='top')

        ax.tick_params(labelsize=fontsize - 1)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.set_ylim(bottom=0)
        if site_type == 'FCC':
            ax.set_xlim(fcc_xmin, fcc_xmax)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_label_coords(-0.15, 0.5)
        ax.xaxis.set_label_coords(0.5, -0.14)

plt.tight_layout()
plt.savefig('figure7_combined.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'figure7_combined.png'")
