#!/usr/bin/env python3
"""
NEP vs DFT 能量-体积对比图绘制脚本

从指定工作目录中的energy_test.out和test.xyz文件读取数据，
绘制NEP预测值与DFT参考值的能量-体积关系对比图。

用法:
    uv run python plot_energy_volume_comparison.py <工作目录>
    
示例:
    uv run python plot_energy_volume_comparison.py nep89_multi-stages_plan1/1/
"""

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_lattice_string(lattice_str):
    """解析Lattice字符串，返回3x3矩阵"""
    match = re.search(r'"([^"]*)"', lattice_str)
    if not match:
        raise ValueError(f"无法解析Lattice字符串: {lattice_str}")
    
    values = list(map(float, match.group(1).split()))
    if len(values) != 9:
        raise ValueError(f"Lattice参数数量错误: {len(values)}, 期望9个")
    
    lattice = np.array(values).reshape(3, 3)
    return lattice


def calculate_volume(lattice_matrix):
    """计算晶胞体积"""
    return abs(np.linalg.det(lattice_matrix))


def parse_xyz_structure(lines, start_idx):
    """解析单个XYZ结构"""
    if start_idx >= len(lines):
        return None, start_idx
    
    try:
        # 读取原子数
        num_atoms = int(lines[start_idx].strip())
        
        # 读取属性行
        properties_line = lines[start_idx + 1].strip()
        
        # 提取config_type
        config_type_match = re.search(r'Config_type=(\w+)', properties_line)
        config_type = config_type_match.group(1) if config_type_match else "unknown"
        
        # 提取能量
        energy_match = re.search(r'Energy=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', properties_line)
        energy = float(energy_match.group(1)) if energy_match else 0.0
        
        # 解析Lattice
        lattice_matrix = parse_lattice_string(properties_line)
        volume = calculate_volume(lattice_matrix)
        # 计算原子单位体积 (Ų/atom)
        volume_per_atom = volume / num_atoms
        
        structure = {
            'num_atoms': num_atoms,
            'config_type': config_type,
            'energy': energy,
            'volume': volume,
            'volume_per_atom': volume_per_atom,
            'energy_per_atom': energy / num_atoms
        }
        
        return structure, start_idx + 2 + num_atoms
        
    except Exception as e:
        print(f"解析结构时出错 (行 {start_idx}): {e}")
        return None, start_idx + 1


def read_test_xyz(filename):
    """读取test.xyz文件并解析所有结构"""
    print(f"读取测试集文件: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    structures = []
    idx = 0
    
    while idx < len(lines):
        structure, idx = parse_xyz_structure(lines, idx)
        if structure:
            structures.append(structure)
    
    print(f"成功解析 {len(structures)} 个结构")
    return structures


def read_energy_test_out(filename):
    """读取energy_test.out文件"""
    print(f"读取能量测试结果: {filename}")
    
    nep_energies = []
    dft_energies = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            try:
                parts = line.split()
                if len(parts) >= 2:
                    nep_energy = float(parts[0])
                    dft_energy = float(parts[1])
                    nep_energies.append(nep_energy)
                    dft_energies.append(dft_energy)
            except ValueError as e:
                print(f"警告: 第{line_num}行数据格式错误: {line}")
                continue
    
    print(f"成功读取 {len(nep_energies)} 对能量数据")
    return nep_energies, dft_energies


def combine_data(structures, nep_energies, dft_energies):
    """合并结构信息和能量数据"""
    if len(structures) != len(nep_energies) or len(structures) != len(dft_energies):
        print(f"警告: 数据长度不匹配!")
        print(f"  结构数: {len(structures)}")
        print(f"  NEP能量数: {len(nep_energies)}")
        print(f"  DFT能量数: {len(dft_energies)}")
        
        # 取最小长度
        min_len = min(len(structures), len(nep_energies), len(dft_energies))
        structures = structures[:min_len]
        nep_energies = nep_energies[:min_len]
        dft_energies = dft_energies[:min_len]
        print(f"  使用前 {min_len} 个数据点")
    
    combined_data = []
    for i, struct in enumerate(structures):
        data_point = {
            'config_type': struct['config_type'],
            'volume_per_atom': struct['volume_per_atom'],
            'nep_energy_per_atom': nep_energies[i],  # 已经是原子平均能量
            'dft_energy_per_atom': dft_energies[i]   # 已经是原子平均能量
        }
        combined_data.append(data_point)
    
    return combined_data


def plot_energy_volume_comparison(data, output_filename, folder_name=None):
    """绘制NEP vs DFT能量-体积对比图"""
    
    # 按晶胞类型分组数据
    data_by_type = defaultdict(list)
    
    for point in data:
        config_type = point['config_type']
        volume_per_atom = point['volume_per_atom']
        nep_energy = point['nep_energy_per_atom']
        dft_energy = point['dft_energy_per_atom']
        
        data_by_type[config_type].append((volume_per_atom, nep_energy, dft_energy))
    
    # 设置图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 定义颜色和标记样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # 左图: NEP vs DFT 能量-体积曲线
    for i, (config_type, data_points) in enumerate(sorted(data_by_type.items())):
        if not data_points:
            continue
        
        # 按体积排序
        data_points.sort(key=lambda x: x[0])
        
        volumes = [point[0] for point in data_points]
        nep_energies = [point[1] for point in data_points]
        dft_energies = [point[2] for point in data_points]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # NEP数据点和连线
        ax1.scatter(volumes, nep_energies, 
                   color=color, marker=marker, s=60, alpha=0.8,
                   label=f'{config_type} NEP ({len(data_points)})')
        if len(data_points) > 1:
            ax1.plot(volumes, nep_energies, color=color, alpha=0.6, linewidth=1.5, linestyle='-')
        
        # DFT数据点和连线
        ax1.scatter(volumes, dft_energies, 
                   color=color, marker=marker, s=60, alpha=0.5, facecolors='none', edgecolors=color,
                   label=f'{config_type} DFT ({len(data_points)})')
        if len(data_points) > 1:
            ax1.plot(volumes, dft_energies, color=color, alpha=0.4, linewidth=1.5, linestyle='--')
    
    ax1.set_xlabel('Volume per atom (Ų/atom)', fontsize=12)
    ax1.set_ylabel('Energy per atom (eV/atom)', fontsize=12)
    title_prefix = f'{folder_name}: ' if folder_name else ''
    ax1.set_title(f'{title_prefix}NEP vs DFT: Energy-Volume Curves', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 右图: NEP vs DFT 散点对比
    all_nep = []
    all_dft = []
    
    for config_type, data_points in sorted(data_by_type.items()):
        if not data_points:
            continue
        
        nep_energies = [point[1] for point in data_points]
        dft_energies = [point[2] for point in data_points]
        
        all_nep.extend(nep_energies)
        all_dft.extend(dft_energies)
        
        color = colors[list(sorted(data_by_type.keys())).index(config_type) % len(colors)]
        marker = markers[list(sorted(data_by_type.keys())).index(config_type) % len(markers)]
        
        ax2.scatter(dft_energies, nep_energies, 
                   color=color, marker=marker, s=60, alpha=0.7,
                   label=f'{config_type} ({len(data_points)})')
    
    # 添加完美预测线 (y=x)
    if len(all_dft) > 0 and len(all_nep) > 0:
        min_energy = min(min(all_dft), min(all_nep))
        max_energy = max(max(all_dft), max(all_nep))
        ax2.plot([min_energy, max_energy], [min_energy, max_energy], 
                'k--', alpha=0.5, linewidth=1, label='Perfect prediction')
        
        # 计算统计指标
        all_nep = np.array(all_nep)
        all_dft = np.array(all_dft)
        
        mae = np.mean(np.abs(all_nep - all_dft))
        rmse = np.sqrt(np.mean((all_nep - all_dft)**2))
        r2 = np.corrcoef(all_nep, all_dft)[0, 1]**2
        
        # 在图上显示统计信息
        stats_text = f'MAE: {mae:.4f} eV/atom\nRMSE: {rmse:.4f} eV/atom\nR²: {r2:.4f}'
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_xlabel('DFT Energy per atom (eV/atom)', fontsize=12)
    ax2.set_ylabel('NEP Energy per atom (eV/atom)', fontsize=12)
    title_prefix = f'{folder_name}: ' if folder_name else ''
    ax2.set_title(f'{title_prefix}NEP vs DFT: Energy Correlation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {output_filename}")
    
    # 显示统计信息
    print("\n数据统计:")
    print("=" * 80)
    print(f"{'晶胞类型':<20} {'结构数':<8} {'体积范围 (Ų/atom)':<25} {'能量范围 (eV/atom)'}")
    print("=" * 80)
    
    for config_type, data_points in sorted(data_by_type.items()):
        if not data_points:
            continue
        
        volumes = [point[0] for point in data_points]
        dft_energies = [point[2] for point in data_points]
        
        vol_min, vol_max = min(volumes), max(volumes)
        energy_min, energy_max = min(dft_energies), max(dft_energies)
        
        print(f"{config_type:<20} {len(data_points):<8} "
              f"{vol_min:.2f} - {vol_max:.2f}        "
              f"{energy_min:.3f} - {energy_max:.3f}")
    
    if len(all_dft) > 0 and len(all_nep) > 0:
        print(f"\n整体预测精度:")
        print(f"  MAE:  {mae:.4f} eV/atom")
        print(f"  RMSE: {rmse:.4f} eV/atom") 
        print(f"  R²:   {r2:.4f}")
    
    return plt


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: uv run python plot_energy_volume_comparison.py <工作目录>")
        print("示例: uv run python plot_energy_volume_comparison.py nep89_multi-stages_plan1/1/")
        sys.exit(1)
    
    work_dir = sys.argv[1].rstrip('/')
    
    # 构建文件路径
    test_xyz_file = os.path.join(work_dir, 'test.xyz')
    energy_test_file = os.path.join(work_dir, 'energy_test.out')
    output_file = os.path.join(work_dir, 'energy_volume_comparison.png')
    
    # 检查文件是否存在
    if not os.path.exists(test_xyz_file):
        print(f"错误: 找不到文件 {test_xyz_file}")
        sys.exit(1)
    
    if not os.path.exists(energy_test_file):
        print(f"错误: 找不到文件 {energy_test_file}")
        sys.exit(1)
    
    try:
        print("=" * 80)
        print(f"处理工作目录: {work_dir}")
        print("=" * 80)
        
        # 读取数据
        print("\n步骤 1: 读取测试集结构信息")
        structures = read_test_xyz(test_xyz_file)
        
        print("\n步骤 2: 读取能量测试结果")
        nep_energies, dft_energies = read_energy_test_out(energy_test_file)
        
        print("\n步骤 3: 合并数据")
        combined_data = combine_data(structures, nep_energies, dft_energies)
        
        print("\n步骤 4: 绘制对比图")
        # 提取文件夹名称
        folder_name = os.path.basename(work_dir.rstrip('/'))
        plt = plot_energy_volume_comparison(combined_data, output_file, folder_name)
        
        print(f"\n" + "=" * 80)
        print("任务完成!")
        print("=" * 80)
        print(f"能量-体积对比图已保存到: {output_file}")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
