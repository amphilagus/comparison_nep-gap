import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Read summary statistics
summary = pd.read_csv('distance_distributions_summary.csv')

# Read zone data files
zones = {
    'Zone 0-25 Å': pd.read_csv('distance_distributions_Zone_0-25_A.csv'),
    'Zone 25-35 Å': pd.read_csv('distance_distributions_Zone_25-35_A.csv'),
    'Zone 35-45 Å': pd.read_csv('distance_distributions_Zone_35-45_A.csv'),
    'Zone 45-60 Å': pd.read_csv('distance_distributions_Zone_45-60_A.csv')
}

# 设置基础尺寸
figsize = 14
fontsize = 9

# 设置字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 使用超精细网格划分方式
n = 100
x0 = 10*n
y0 = 8*n
dy = int(y0/10)
dx = int(x0/5)
M = int(3*x0 + 2*dx)
N = int(4*y0 + 3*dy)

# 创建图形
fig = plt.figure(figsize=(figsize, N/(M/figsize)))
gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))

# 颜色设置
colors = {
    'FCC': '#5B8DBE',      # 蓝色 for O atoms to FCC
    '4-coord': '#6EAA5E',   # 绿色 for Ga to 4-coord
    '6-coord': '#E6955C'    # 橙色 for Ga to 6-coord
}

# 列定义
columns = [
    ('FCC', 'O Atoms to FCC Sites', 'O'),
    ('4-coord', 'Ga Atoms to 4-Coordinated Sites', 'Ga'),
    ('6-coord', 'Ga Atoms to 6-Coordinated Sites', 'Ga')
]

# 行定义
zone_names = ['Zone 0-25 Å', 'Zone 25-35 Å', 'Zone 35-45 Å', 'Zone 45-60 Å']
real_names = ['Region I', 'Region II', 'Region III', 'Region IV']
# 计算 cutoff (mean + 3*std)
def calculate_cutoff(mean, std):
    return mean + 3 * std

# 计算 within cutoff 百分比
def calculate_within_cutoff_percentage(data, cutoff):
    within = np.sum(data <= cutoff)
    total = len(data)
    return (within / total) * 100 if total > 0 else 0

# 绘制子图
for row_idx, zone_name in enumerate(zone_names):
    zone_data = zones[zone_name]
    real_name = real_names[row_idx]

    for col_idx, (site_type, title, atom_type) in enumerate(columns):
        # 创建子图
        ax = fig.add_subplot(gs[row_idx*(y0+dy):row_idx*(y0+dy)+y0, 
                                col_idx*(x0+dx):col_idx*(x0+dx)+x0])
        
        # 筛选数据
        data = zone_data[zone_data['site_type'] == site_type]['distance_angstrom'].values
        
        # 获取统计信息
        stats = summary[(summary['zone'] == zone_name) & 
                       (summary['site_type'] == site_type)]
        
        if len(stats) > 0 and len(data) > 0:
            mean = stats['mean'].values[0]
            std = stats['std'].values[0]
            count = stats['count'].values[0]
            ratio = stats['ratio'].values[0]
            
            # 绘制直方图
            n_bins = 50
            ax.hist(data, bins=n_bins, color=colors[site_type], 
                   alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 添加均值线
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean:.4f} Å')
            
            # 准备统计信息文本
            textstr = f'Mean = {mean:.4f} Å'
            
            if atom_type == 'Ga':
                cutoff = 0.7550
                within_pct = ratio * 100
                
                # Add cutoff line
                ax.axvline(cutoff, color='green', linestyle=':', linewidth=2)
                textstr += f'\nWithin cutoff: {within_pct:.1f}%'
            
            # 添加统计信息框
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.95 if col_idx == 0 else 0.05, 0.95, textstr, transform=ax.transAxes,
                   fontsize=fontsize-1, verticalalignment='top',
                   horizontalalignment='right' if col_idx == 0 else 'left', bbox=props)
        
        # 设置标签
        ax.set_xlabel('Distance to Nearest ' + 
                     ('FCC Site' if site_type == 'FCC' else 
                      f'{site_type.split("-")[0]}-Coordinate Site') + ' (Å)', 
                     fontsize=fontsize)
        ax.set_ylabel('Count', fontsize=fontsize)
        
        # 设置标题（只在第一行显示）
        if row_idx == 0:
            ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=8)
        
        # 添加区域标签（只在第一列显示）
        if col_idx == 0:
            # 在左侧添加区域标签
            ax.text(-0.2, 0.5, real_name, transform=ax.transAxes,
                   fontsize=fontsize+1, fontweight='bold',
                   verticalalignment='center', horizontalalignment='center',
                   rotation=90)
        
        # 添加子图标签
        label_idx = row_idx * len(columns) + col_idx
        label = f'({chr(97 + label_idx)})'
        ax.text(-0.15, 1.05, label, transform=ax.transAxes,
               fontsize=fontsize+4, fontweight='bold', ha='left', va='top')
        
        # 设置刻度
        ax.tick_params(labelsize=fontsize-1)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # 设置y轴从0开始
        ax.set_ylim(bottom=0)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('distance_distributions.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'distance_distributions.png'")

# plt.show()
