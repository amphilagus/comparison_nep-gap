import MDemon as md
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import numpy as np

u = md.Universe('cropped.xyz')

coordinates = u.atoms.coordinate

# Calculate center of mass and shift coordinates to center at origin
center_of_mass = np.mean(coordinates, axis=0)
coordinates = coordinates - center_of_mass

element_dic = {1:"Ga", 2:"O"}
elements = [element_dic[i] for i in u.atoms.species]

# 分离 Ga 和 O 原子的坐标
ga_coords = coordinates[np.array(elements) == 'Ga']
o_coords = coordinates[np.array(elements) == 'O']

# 设置基础尺寸
figsize = 10
fontsize = 10

# 设置字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 使用超精细网格划分方式
n = 100
x0 = 10*n
x1 = int(6*n*10/10.5)
y0 = 10*n  # 使用正方形子图
dy = int(y0/10)
dx = int(x0/10)
M = int(x0 + x1 + dx)
N = int(y0 + 2*dy)

# 创建图形
fig = plt.figure(figsize=(figsize, N/(M/figsize)))
gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))

# 颜色设置
color_ga = '#0d47a1'  # 深蓝色 for Ga
color_o = '#d84315'   # 深橙色 for O
marker_size = 1.0
alpha = 0.8

# 子图 1: 横截面 (Transverse Section - XZ plane, y plane)
ax1 = fig.add_subplot(gs[dy:dy+y0, 0:x0])
ax1.scatter(ga_coords[:, 0], ga_coords[:, 2], s=marker_size, alpha=alpha, 
           color=color_ga, label='Ga', edgecolor='none')
ax1.scatter(o_coords[:, 0], o_coords[:, 2], s=marker_size*0.6, alpha=alpha, 
           color=color_o, label='O', edgecolor='none')
ax1.set_xlabel('X (Å)', fontsize=fontsize)
ax1.set_ylabel('Z (Å)', fontsize=fontsize)
ax1.legend(loc='upper right', fontsize=fontsize-1, markerscale=5)
ax1.tick_params(labelsize=fontsize-1)
ax1.tick_params(axis='y', rotation=90)
# Set y-axis tick labels to center align
for label in ax1.get_yticklabels():
    label.set_verticalalignment('center')
ax1.set_aspect('equal', adjustable='box')
ax1.text(-0.1, 1.05, '(a)', transform=ax1.transAxes,
        fontsize=fontsize+3, fontweight='bold', ha='left', va='top')

# Add three concentric circles
radii = [25, 35, 45]
for radius in radii:
    circle = Circle((0, 0), radius, fill=False, edgecolor='black', 
                   linewidth=1.5, linestyle='--', zorder=5)
    ax1.add_patch(circle)

# Add Roman numerals for the four regions
# Region I: r < 28 (center)
ax1.text(0, 0, 'I', fontsize=fontsize+4, fontweight='bold', 
        ha='center', va='center', color='black', zorder=6)

# Region II: 28 < r < 38
angle_ii = np.pi/4  # 45 degrees
r_ii = (25 + 35) / 2
ax1.text(r_ii*np.cos(angle_ii), r_ii*np.sin(angle_ii), 'II', 
        fontsize=fontsize+4, fontweight='bold', 
        ha='center', va='center', color='black', zorder=6)

# Region III: 38 < r < 48
angle_iii = np.pi/4  # 45 degrees
r_iii = (35 + 45) / 2
ax1.text(r_iii*np.cos(angle_iii), r_iii*np.sin(angle_iii), 'III', 
        fontsize=fontsize+4, fontweight='bold', 
        ha='center', va='center', color='black', zorder=6)

# Region IV: r > 48
angle_iv = np.pi/4  # 45 degrees
r_iv = 50
ax1.text(r_iv*np.cos(angle_iv), r_iv*np.sin(angle_iv), 'IV', 
        fontsize=fontsize+4, fontweight='bold', 
        ha='center', va='center', color='black', zorder=6)

ax1.set_xlim(-60, 60)
ax1.set_ylim(-60, 60)

# 子图 2: 纵截面 (Longitudinal Section - XY plane)
ax2 = fig.add_subplot(gs[dy:dy+y0, x0+dx:x0+dx+x1])
ax2.scatter(ga_coords[:, 0], ga_coords[:, 1], s=marker_size, alpha=alpha, 
           color=color_ga, label='Ga', edgecolor='none')
ax2.scatter(o_coords[:, 0], o_coords[:, 1], s=marker_size*0.6, alpha=alpha, 
           color=color_o, label='O', edgecolor='none')
ax2.set_xlabel('X (Å)', fontsize=fontsize)
ax2.set_ylabel('Y (Å)', fontsize=fontsize)
# ax2.legend(loc='upper right', fontsize=fontsize-1, markerscale=5)
ax2.tick_params(labelsize=fontsize-1)
ax2.tick_params(axis='y', rotation=90)
# Set y-axis tick labels to center align
for label in ax2.get_yticklabels():
    label.set_verticalalignment('center')
ax2.set_aspect('equal', adjustable='box')
ax2.text(-0.1, 1.05, '(b)', transform=ax2.transAxes,
        fontsize=fontsize+3, fontweight='bold', ha='left', va='top')
ax2.set_xlim(-60, 60)
ax2.set_ylim(-105, 105)
# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('transverse_longitudinal_sections.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'transverse_longitudinal_sections.png'")

# plt.show()