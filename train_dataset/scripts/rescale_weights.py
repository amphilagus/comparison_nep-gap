#!/usr/bin/env python3
"""
根据平均能量rescale权重的脚本

用法: python rescale_weights.py <xyz_file> <alpha>
"""

import sys
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_xyz_file(filename):
    """解析xyz文件，返回结构列表"""
    structures = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # 读取原子数
        try:
            n_atoms = int(lines[i].strip())
        except (ValueError, IndexError):
            break
        
        # 读取配置行
        config_line = lines[i + 1]
        
        # 提取能量和权重（不区分大小写）
        energy_match = re.search(r'[Ee]nergy=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', config_line)
        weight_match = re.search(r'[Ww]eight=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', config_line)
        
        # 提取config_type（不区分大小写）
        config_type_match = re.search(r'[Cc]onfig_[Tt]ype=(\S+)', config_line)
        config_type = config_type_match.group(1) if config_type_match else 'unknown'
        
        if energy_match and weight_match:
            energy = float(energy_match.group(1))
            weight = float(weight_match.group(1))
            
            # 计算平均能量
            avg_energy = energy / n_atoms
            
            # 保存结构信息
            structure = {
                'n_atoms': n_atoms,
                'config_line': config_line,
                'atom_lines': lines[i + 2:i + 2 + n_atoms],
                'energy': energy,
                'weight': weight,
                'avg_energy': avg_energy,
                'config_type': config_type
            }
            structures.append(structure)
        
        # 移动到下一个结构
        i += 2 + n_atoms
    
    return structures


def rescale_weights(structures, alpha):
    """根据平均能量rescale权重"""
    # 找到最小平均能量
    min_avg_energy = min(s['avg_energy'] for s in structures)
    
    print(f"最小平均能量: {min_avg_energy:.6f} eV/atom")
    print(f"Rescale幂指数 alpha: {alpha}")
    print(f"总结构数: {len(structures)}")
    print()
    
    # 计算新权重
    for structure in structures:
        e = structure['avg_energy']
        e1 = min_avg_energy
        
        # rescale比例 = 1 / (1 + (e - e1)^alpha)
        rescale_factor = 1.0 / (1.0 + (e - e1) ** alpha)
        
        # 新权重 = 旧权重 * rescale比例
        new_weight = structure['weight'] * rescale_factor
        structure['new_weight'] = new_weight
        structure['rescale_factor'] = rescale_factor
    
    return structures


def write_rescaled_xyz(structures, output_filename):
    """写入rescale后的xyz文件"""
    with open(output_filename, 'w') as f:
        for structure in structures:
            # 写入原子数
            f.write(f"{structure['n_atoms']}\n")
            
            # 更新配置行中的权重（保持原有大小写格式）
            config_line = structure['config_line']
            new_config_line = re.sub(
                r'([Ww]eight=)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                lambda m: f"{m.group(1)}{structure['new_weight']:.10f}",
                config_line
            )
            f.write(new_config_line)
            
            # 写入原子坐标
            for atom_line in structure['atom_lines']:
                f.write(atom_line)
    
    print(f"已写入rescale后的文件: {output_filename}")


def plot_rescale_analysis(structures, output_filename, alpha):
    """绘制权重缩放分析图"""
    # 提取数据
    energies = [s['avg_energy'] for s in structures]
    rescale_factors = [s['rescale_factor'] for s in structures]
    config_types = [s['config_type'] for s in structures]
    
    # 获取所有唯一的config_type并排序
    unique_config_types = sorted(list(set(config_types)))
    config_type_to_idx = {ct: idx for idx, ct in enumerate(unique_config_types)}
    
    # 创建颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_config_types)))
    color_map = {ct: colors[i] for i, ct in enumerate(unique_config_types)}
    
    # 创建图形
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 左侧y轴：权重缩放系数
    ax1.set_xlabel('Average Energy (eV/atom)', fontsize=12)
    ax1.set_ylabel('Rescale Factor', fontsize=12, color='black')
    
    # 绘制理论曲线（rescale factor vs energy）
    min_energy = min(energies)
    max_energy = max(energies)
    energy_range = np.linspace(min_energy, max_energy, 500)
    rescale_curve = 1.0 / (1.0 + (energy_range - min_energy) ** alpha)
    ax1.plot(energy_range, rescale_curve, 'k-', linewidth=2, zorder=1)
    
    # 设置左侧y轴为对数轴
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # 右侧twin y轴：config_type索引
    ax2 = ax1.twinx()
    ax2.set_ylabel('Config Type', fontsize=12, color='gray')
    
    # 在右侧y轴上绘制config_type散点（使用相同的颜色）
    for ct in unique_config_types:
        mask = [config_types[i] == ct for i in range(len(config_types))]
        e_filtered = [energies[i] for i in range(len(energies)) if mask[i]]
        ct_idx = config_type_to_idx[ct]
        ct_indices = [ct_idx] * len(e_filtered)
        ax2.scatter(e_filtered, ct_indices, c=[color_map[ct]], alpha=0.6, s=30, zorder=2)
    
    ax2.set_yticks(range(len(unique_config_types)))
    ax2.set_yticklabels(unique_config_types, fontsize=9)
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # 添加标题
    plt.title(f'Weight Rescale Factor vs Energy by Config Type (α={alpha})', fontsize=14, pad=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plot_filename = output_filename.replace('.xyz', '.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"已保存分析图: {plot_filename}")
    
    plt.close()


def main():
    if len(sys.argv) != 3:
        print("用法: python rescale_weights.py <xyz_file> <alpha>")
        print("示例: python rescale_weights.py train.xyz 2.0")
        sys.exit(1)
    
    xyz_file = sys.argv[1]
    alpha = float(sys.argv[2])
    
    # 生成输出文件名
    if xyz_file.endswith('.xyz'):
        output_file = xyz_file[:-4] + f'_rescaled-{alpha}.xyz'
    else:
        output_file = xyz_file + f'_rescaled-{alpha}.xyz'
    
    print(f"输入文件: {xyz_file}")
    print(f"输出文件: {output_file}")
    print()
    
    # 解析xyz文件
    print("正在解析xyz文件...")
    structures = parse_xyz_file(xyz_file)
    
    # Rescale权重
    print("正在rescale权重...")
    structures = rescale_weights(structures, alpha)
    
    # 写入新文件
    write_rescaled_xyz(structures, output_file)
    
    # 绘制分析图
    print()
    print("正在生成分析图...")
    plot_rescale_analysis(structures, output_file, alpha)
    
    # 打印统计信息
    print()
    print("权重统计:")
    print(f"  原始权重范围: {min(s['weight'] for s in structures):.6f} - {max(s['weight'] for s in structures):.6f}")
    print(f"  新权重范围: {min(s['new_weight'] for s in structures):.6f} - {max(s['new_weight'] for s in structures):.6f}")
    print(f"  Rescale因子范围: {min(s['rescale_factor'] for s in structures):.6f} - {max(s['rescale_factor'] for s in structures):.6f}")


if __name__ == '__main__':
    main()
