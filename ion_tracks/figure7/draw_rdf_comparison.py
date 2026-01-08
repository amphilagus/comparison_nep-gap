import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Read CSV files
rdf_ga_ga = pd.read_csv('rdf_Ga_Ga.csv')
rdf_ga_o = pd.read_csv('rdf_Ga_O.csv')
rdf_o_o = pd.read_csv('rdf_O_O.csv')

# 设置基础尺寸
figsize = 14
fontsize = 10

# 设置字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 使用超精细网格划分方式
n = 100
x0 = 10*n
y0 = 8*n
dy = int(y0/20)
dx = int(x0/5)
M = int(3*x0 + 2*dx)
N = int(y0 + 2*dy)

# 创建图形
fig = plt.figure(figsize=(figsize, N/(M/figsize)))
gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))

# 颜色和线型设置
colors = {
    'pristine': '#1565c0',  # 深蓝色
    'core': '#d32f2f',      # 红色
    'shell': '#f57c00',     # 橙色
    'outer': '#00897b',     # 青色
    'far': '#7b1fa2'        # 紫色
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

# 数据集和标题
datasets = [
    (rdf_ga_ga, 'Ga-Ga RDF'),
    (rdf_ga_o, 'Ga-O RDF'),
    (rdf_o_o, 'O-O RDF')
]

# 子图标签
subplot_labels = ['(a)', '(b)', '(c)']

# 绘制三个子图
for idx, (data, title) in enumerate(datasets):
    ax = fig.add_subplot(gs[dy:dy+y0, idx*(x0+dx):idx*(x0+dx)+x0])
    
    # 提取距离数据
    r = data['r_angstrom']
    
    # 绘制各条曲线
    ax.plot(r, data['pristine_rdf'], 
            color=colors['pristine'], 
            linestyle=linestyles['pristine'],
            linewidth=linewidths['pristine'],
            label='Pristine')
    
    ax.plot(r, data['Irr._Core_0-15A_rdf'], 
            color=colors['core'], 
            linestyle=linestyles['core'],
            linewidth=linewidths['core'],
            label=r'Irr. Core (0-15$\,$Å)')
    
    ax.plot(r, data['Irr._Shell_15-30A_rdf'], 
            color=colors['shell'], 
            linestyle=linestyles['shell'],
            linewidth=linewidths['shell'],
            label=r'Irr. Shell (15-30$\,$Å)')
    
    ax.plot(r, data['Irr._Outer_30-45A_rdf'], 
            color=colors['outer'], 
            linestyle=linestyles['outer'],
            linewidth=linewidths['outer'],
            label=r'Irr. Outer (30-45$\,$Å)')
    
    ax.plot(r, data['Irr._Far_45-60A_rdf'], 
            color=colors['far'], 
            linestyle=linestyles['far'],
            linewidth=linewidths['far'],
            label=r'Irr. Far (45-60$\,$Å)')
    
    # 设置标签和标题
    ax.set_xlabel(r'$r\ $(Å)', fontsize=fontsize)
    ax.set_ylabel(r'$g(r)$', fontsize=fontsize)
    
    # 设置刻度
    ax.tick_params(labelsize=fontsize-1)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 添加图例
    if idx == 0:
        ax.legend(loc='upper left', fontsize=fontsize-2, framealpha=0.9)
    else:
        pass
        # ax.legend(loc='upper left', fontsize=fontsize-2, framealpha=0.9)
    
    # 添加子图标签
    ax.text(-0.08, 1.05, subplot_labels[idx], transform=ax.transAxes,
           fontsize=fontsize+4, fontweight='bold', ha='left', va='top')
    
    # 设置y轴从0开始
    ax.set_ylim(bottom=0)
    
    # 设置x轴范围
    ax.set_xlim(1.0, 5.5)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片
plt.savefig('rdf_comparison.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'rdf_comparison.png'")

# plt.show()
