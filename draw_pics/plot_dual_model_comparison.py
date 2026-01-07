#!/usr/bin/env python3
"""
集成的双模型对比分析脚本
自动校准NEP能量基线到tabGAP，并绘制2列3行的对比图
只需指定NEP版本号和数据集名称，自动推断所有文件路径
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import re
from typing import Dict, List, Tuple


class EnergyAligner:
    """能量基线校准工具"""
    def __init__(self):
        # tabGAP zero-point energies (eV)
        self.tabgap_zpe = {
            'Ga': -0.0244486,
            'O': -0.0350174
        }
        
        # NEP zero-point energies (eV)
        self.nep_zpe = {
            'Ga': -1.68768,
            'O': -3.19589
        }
    
    def count_atoms_in_xyz(self, xyz_file: str) -> Dict[str, int]:
        """从xyz文件中统计原子数"""
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return {'Ga': 0, 'O': 0}
        
        n_atoms = int(lines[0].strip())
        atom_counts = {'Ga': 0, 'O': 0}
        
        for i in range(2, min(2 + n_atoms, len(lines))):
            parts = lines[i].strip().split()
            if len(parts) >= 1:
                element = parts[0]
                if element in atom_counts:
                    atom_counts[element] += 1
        
        return atom_counts
    
    def calculate_offset_per_atom(self, atom_counts: Dict[str, int]) -> float:
        """
        计算从NEP基线到tabGAP基线的per-atom能量偏移
        
        Args:
            atom_counts: 原子计数 {'Ga': n_Ga, 'O': n_O}
        
        Returns:
            per-atom能量偏移量（eV/atom）
        """
        n_total = atom_counts['Ga'] + atom_counts['O']
        if n_total == 0:
            return 0.0
        
        # 计算总偏移量（从NEP到tabGAP）
        offset_total = (
            atom_counts['Ga'] * (self.tabgap_zpe['Ga'] - self.nep_zpe['Ga']) +
            atom_counts['O'] * (self.tabgap_zpe['O'] - self.nep_zpe['O'])
        )
        
        # 转换为per-atom偏移
        offset_per_atom = offset_total / n_total
        
        return offset_per_atom


def parse_xyz_properties(xyz_file: str) -> Dict:
    """从xyz文件中提取DFT参考值"""
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        raise ValueError(f"文件 {xyz_file} 格式不正确")
    
    n_atoms = int(lines[0].strip())
    header_line = lines[1].strip()
    
    # 提取virial (eV)
    virial_match = re.search(r'virial="([^"]*)"', header_line, re.IGNORECASE)
    if virial_match:
        virial_str = virial_match.group(1)
        virial_values = [float(x) for x in virial_str.split()]
        
        if len(virial_values) == 9:
            virial = [
                virial_values[0],  # xx
                virial_values[4],  # yy
                virial_values[8],  # zz
                virial_values[1],  # xy
                virial_values[2],  # xz
                virial_values[5],  # yz
            ]
            virial_per_atom = [v / n_atoms for v in virial]
        else:
            virial_per_atom = None
    else:
        virial_per_atom = None
    
    # 提取力 (eV/Å)
    forces = []
    for i in range(2, 2 + n_atoms):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        if len(parts) >= 7:
            fx, fy, fz = float(parts[4]), float(parts[5]), float(parts[6])
            forces.append([fx, fy, fz])
    
    forces = np.array(forces) if forces else None
    
    return {
        'virial_per_atom': virial_per_atom,
        'forces': forces,
    }


def parse_lammps_forces(dump_file: str) -> np.ndarray:
    """从dump.forces文件中提取力"""
    data = []
    with open(dump_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('ITEM:'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                fx, fy, fz = float(parts[5]), float(parts[6]), float(parts[7])
                data.append([fx, fy, fz])
    return np.array(data)


def parse_lammps_summary(summary_file: str) -> Dict:
    """从summary.txt文件中提取LAMMPS计算结果"""
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
        
        result = {}
        
        # 提取virial分量 (eV/atom)
        virial_pattern = r'Average virial per atom \(eV/atom\):\s*vxx\s*=\s*([-\d.]+)\s*vyy\s*=\s*([-\d.]+)\s*vzz\s*=\s*([-\d.]+)\s*vxy\s*=\s*([-\d.]+)\s*vxz\s*=\s*([-\d.]+)\s*vyz\s*=\s*([-\d.]+)'
        virial_match = re.search(virial_pattern, content)
        if virial_match:
            vxx = float(virial_match.group(1))
            vyy = float(virial_match.group(2))
            vzz = float(virial_match.group(3))
            vxy = float(virial_match.group(4))
            vxz = float(virial_match.group(5))
            vyz = float(virial_match.group(6))
            result['virial'] = [vxx, vyy, vzz, vxy, vxz, vyz]
        else:
            result['virial'] = None
        
        return result
    except Exception as e:
        return {'virial': None}


def align_nep_energies(nep_csv: str, raw_data_dir: str, tabgap_csv: str) -> Tuple[List[Dict], bool]:
    """
    校准NEP能量到tabGAP基线
    
    Returns:
        (aligned_data, success): 校准后的数据和是否成功的标志
    """
    # 读取NEP CSV
    nep_data = []
    with open(nep_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nep_data.append(row)
    
    # 读取tabGAP CSV用于验证
    tabgap_dft_energies = {}
    with open(tabgap_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tabgap_dft_energies[row['Structure_ID']] = float(row['DFT_Energy_eV_per_atom'])
    
    # 初始化能量校准器
    aligner = EnergyAligner()
    raw_path = Path(raw_data_dir)
    
    print("\n" + "=" * 60)
    print("NEP能量基线校准到tabGAP")
    print("=" * 60)
    print(f"NEP零点能量: Ga={aligner.nep_zpe['Ga']:.6f} eV, O={aligner.nep_zpe['O']:.6f} eV")
    print(f"tabGAP零点能量: Ga={aligner.tabgap_zpe['Ga']:.6f} eV, O={aligner.tabgap_zpe['O']:.6f} eV")
    print("=" * 60)
    
    # 处理每一行数据
    aligned_data = []
    success_count = 0
    fail_count = 0
    dft_diffs = []
    
    print("\n正在校准能量...")
    for i, row in enumerate(nep_data):
        structure_id = row['Structure_ID']
        
        # 找到对应的xyz文件
        xyz_file = raw_path / structure_id / "structure.xyz"
        
        if not xyz_file.exists():
            fail_count += 1
            aligned_data.append(row)
            continue
        
        try:
            # 统计原子数
            atom_counts = aligner.count_atoms_in_xyz(str(xyz_file))
            
            # 计算偏移量
            offset_per_atom = aligner.calculate_offset_per_atom(atom_counts)
            
            # 应用偏移到DFT和LAMMPS能量
            original_dft = float(row['DFT_Energy_eV_per_atom'])
            original_lammps = float(row['LAMMPS_Energy_eV_per_atom'])
            
            aligned_dft = original_dft + offset_per_atom
            aligned_lammps = original_lammps + offset_per_atom
            
            # 重新计算误差
            abs_error = abs(aligned_lammps - aligned_dft)
            rel_error = (abs_error / abs(aligned_dft) * 100) if aligned_dft != 0 else 0.0
            
            # 创建新行
            new_row = row.copy()
            new_row['DFT_Energy_eV_per_atom'] = f"{aligned_dft:.12f}"
            new_row['LAMMPS_Energy_eV_per_atom'] = f"{aligned_lammps:.12f}"
            new_row['Absolute_Error_eV_per_atom'] = f"{abs_error:.12f}"
            new_row['Relative_Error_percent'] = f"{rel_error:.6f}"
            
            aligned_data.append(new_row)
            
            # 验证：与tabGAP的DFT能量比较
            if structure_id in tabgap_dft_energies:
                tabgap_dft = tabgap_dft_energies[structure_id]
                dft_diff = abs(aligned_dft - tabgap_dft)
                dft_diffs.append(dft_diff)
                
                # 打印前3个示例
                if success_count < 3:
                    print(f"\n  示例 {structure_id}:")
                    print(f"    化学计量比: Ga={atom_counts['Ga']}, O={atom_counts['O']}")
                    print(f"    偏移量: {offset_per_atom:.10f} eV/atom")
                    print(f"    NEP DFT原始: {original_dft:.10f} eV/atom")
                    print(f"    NEP DFT校准后: {aligned_dft:.10f} eV/atom")
                    print(f"    tabGAP DFT: {tabgap_dft:.10f} eV/atom")
                    print(f"    差异: {dft_diff:.6e} eV/atom")
            
            success_count += 1
            
        except Exception as e:
            print(f"  警告: {structure_id} 处理失败 - {str(e)}")
            fail_count += 1
            aligned_data.append(row)
    
    print(f"\n数据处理完成: 成功 {success_count} 个，失败 {fail_count} 个")
    
    # 验证结果
    validation_success = False
    if dft_diffs:
        max_diff = max(dft_diffs)
        mean_diff = sum(dft_diffs) / len(dft_diffs)
        
        print("\n" + "=" * 60)
        print("能量基线校准验证")
        print("=" * 60)
        print(f"与tabGAP DFT能量比较（{len(dft_diffs)}个共同结构）:")
        print(f"  最大差异: {max_diff:.6e} eV/atom")
        print(f"  平均差异: {mean_diff:.6e} eV/atom")
        
        if max_diff < 1e-9:
            print(f"\n  ✓ 完美校准！两个模型的DFT能量完全一致（差异 < 1e-9）")
            validation_success = True
        elif max_diff < 1e-6:
            print(f"\n  ✓ 校准成功！两个模型的DFT能量基本一致（差异 < 1e-6）")
            validation_success = True
        elif max_diff < 1e-3:
            print(f"\n  ○ 校准基本正确（差异在合理范围内）")
            validation_success = True
        else:
            print(f"\n  ⚠ 警告：校准后DFT能量仍有较大差异，请检查零点能量设置")
        print("=" * 60)
    
    return aligned_data, validation_success


def collect_data_from_csv_and_raw(csv_file: str, raw_data_dir: str) -> Tuple[Dict, Dict]:
    """从CSV文件和原始数据目录收集数据"""
    # 读取CSV文件
    csv_data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_data.append(row)
    
    raw_path = Path(raw_data_dir)
    
    dft_data = {
        'energy': [],
        'forces': [],
        'virial': []
    }
    
    lammps_data = {
        'energy': [],
        'forces': [],
        'virial': []
    }
    
    success_count = 0
    fail_count = 0
    
    for row in csv_data:
        structure_id = row['Structure_ID']
        dft_energy = float(row['DFT_Energy_eV_per_atom'])
        lammps_energy = float(row['LAMMPS_Energy_eV_per_atom'])
        
        # 构建对应的原始数据路径
        subdir = raw_path / structure_id
        xyz_file = subdir / "structure.xyz"
        force_file = subdir / "dump.forces"
        summary_file = subdir / "summary.txt"
        
        # 检查必要文件是否存在
        if not xyz_file.exists() or not force_file.exists() or not summary_file.exists():
            fail_count += 1
            continue
        
        try:
            # 解析DFT参考值
            dft_props = parse_xyz_properties(str(xyz_file))
            
            # 解析LAMMPS预测值
            lammps_forces = parse_lammps_forces(str(force_file))
            lammps_summary = parse_lammps_summary(str(summary_file))
            lammps_virial = lammps_summary['virial']
            
            # 添加能量数据
            dft_data['energy'].append(dft_energy)
            lammps_data['energy'].append(lammps_energy)
            
            # 添加力数据
            if dft_props['forces'] is not None and len(lammps_forces) > 0:
                dft_forces_flat = dft_props['forces'].flatten()
                lammps_forces_flat = lammps_forces.flatten()
                dft_data['forces'].extend(dft_forces_flat)
                lammps_data['forces'].extend(lammps_forces_flat)
            
            # 添加virial数据
            if dft_props['virial_per_atom'] is not None and lammps_virial is not None:
                dft_data['virial'].extend(dft_props['virial_per_atom'])
                lammps_data['virial'].extend(lammps_virial)
            
            success_count += 1
            
        except Exception as e:
            fail_count += 1
    
    # 转换为numpy数组
    for key in dft_data:
        dft_data[key] = np.array(dft_data[key])
        lammps_data[key] = np.array(lammps_data[key])
    
    return dft_data, lammps_data


def filter_data_by_threshold(dft_data: Dict, lammps_data: Dict, threshold: float) -> Tuple[Dict, Dict]:
    """根据能量阈值筛选数据"""
    if len(dft_data['energy']) == 0:
        return dft_data, lammps_data
    
    # 找到最低能量
    min_energy = np.min(dft_data['energy'])
    energy_threshold = min_energy + threshold
    
    # 筛选低能量区间的数据
    energy_mask = dft_data['energy'] < energy_threshold
    
    filtered_dft = {}
    filtered_lammps = {}
    
    # 筛选能量数据
    filtered_dft['energy'] = dft_data['energy'][energy_mask]
    filtered_lammps['energy'] = lammps_data['energy'][energy_mask]
    
    # 对于力和virial，按比例随机采样
    n_selected = len(filtered_dft['energy'])
    n_total = len(dft_data['energy'])
    
    if n_total > 0:
        sample_ratio = n_selected / n_total
        
        # 力数据采样
        if len(dft_data['forces']) > 0:
            n_force_samples = int(len(dft_data['forces']) * sample_ratio)
            if n_force_samples > 0:
                indices = np.random.choice(len(dft_data['forces']), n_force_samples, replace=False)
                filtered_dft['forces'] = dft_data['forces'][indices]
                filtered_lammps['forces'] = lammps_data['forces'][indices]
            else:
                filtered_dft['forces'] = np.array([])
                filtered_lammps['forces'] = np.array([])
        else:
            filtered_dft['forces'] = np.array([])
            filtered_lammps['forces'] = np.array([])
        
        # virial数据采样
        if len(dft_data['virial']) > 0:
            n_virial_samples = int(len(dft_data['virial']) * sample_ratio)
            if n_virial_samples > 0:
                indices = np.random.choice(len(dft_data['virial']), n_virial_samples, replace=False)
                filtered_dft['virial'] = dft_data['virial'][indices]
                filtered_lammps['virial'] = lammps_data['virial'][indices]
            else:
                filtered_dft['virial'] = np.array([])
                filtered_lammps['virial'] = np.array([])
        else:
            filtered_dft['virial'] = np.array([])
            filtered_lammps['virial'] = np.array([])
    else:
        filtered_dft['forces'] = np.array([])
        filtered_lammps['forces'] = np.array([])
        filtered_dft['virial'] = np.array([])
        filtered_lammps['virial'] = np.array([])
    
    return filtered_dft, filtered_lammps


def calculate_r2(true, pred):
    """计算决定系数R²"""
    if len(true) == 0:
        return 0.0
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)


def plot_dual_comparison_2x3(dft_data1_t5: Dict, lammps_data1_t5: Dict, 
                            dft_data1_t05: Dict, lammps_data1_t05: Dict,
                            dft_data2_t5: Dict, lammps_data2_t5: Dict,
                            dft_data2_t05: Dict, lammps_data2_t05: Dict,
                            output_file: str,
                            model1_name: str = 'NEP',
                            model2_name: str = 'tabGAP',
                            sparse_ratio: float = 0.1):
    """绘制双模型对比的2列3行图"""
    
    # 设置字体为 Times New Roman
    # plt.rcParams['font.family'] = 'Helvetica'
    
    # 设置图形
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # 添加列标题
    axes[0, 0].set_title("Total", fontsize=24, fontweight='bold', pad=40)
    axes[0, 1].set_title("Low_Energy", fontsize=24, fontweight='bold', pad=40)
    
    # 数据类型和标签
    data_types = ['energy', 'forces', 'virial']
    data_labels = ['Energy (eV/atom)', 'Force (eV/Å)', 'Virial (eV/atom)']
    
    # 模型1和模型2的颜色和标记
    color1 = 'blue'
    color2 = 'red'
    marker1 = 'o'
    marker2 = 'x'
    
    # 数据集
    data_sets1 = [(dft_data1_t5, lammps_data1_t5), (dft_data1_t05, lammps_data1_t05)]
    data_sets2 = [(dft_data2_t5, lammps_data2_t5), (dft_data2_t05, lammps_data2_t05)]
    
    # 子图标题
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    plot_index = 0
    for row in range(3):  # 3行：能量、力、位力
        for col in range(2):  # 2列：t=5.0, t=0.5
            ax = axes[row, col]
            dft_data1, lammps_data1 = data_sets1[col]
            dft_data2, lammps_data2 = data_sets2[col]
            data_type = data_types[row]
            
            # 添加子图标题
            ax.text(-0.1, 1.1, subplot_labels[plot_index], transform=ax.transAxes,
                   fontsize=22, fontweight='bold', ha='left', va='top')
            
            # 收集所有数据点以确定坐标轴范围
            all_dft_vals = []
            all_lammps_vals = []

            alpha = 0.5 if data_type == 'energy' else 0.5
            s = 8 if data_type == 'energy' else 0.3 if data_type == 'forces' else 5          

            # 绘制模型1
            if len(dft_data1[data_type]) > 0:
                dft_vals1 = dft_data1[data_type]
                lammps_vals1 = lammps_data1[data_type]
                
                # 对Force和Virial数据进行稀疏化采样
                if data_type in ['forces', 'virial'] and len(dft_vals1) > 0:
                    n_samples = max(1, int(len(dft_vals1) * sparse_ratio))
                    indices = np.random.choice(len(dft_vals1), n_samples, replace=False)
                    dft_vals1_plot = dft_vals1[indices]
                    lammps_vals1_plot = lammps_vals1[indices]
                else:
                    dft_vals1_plot = dft_vals1
                    lammps_vals1_plot = lammps_vals1
                
                ax.scatter(dft_vals1_plot, lammps_vals1_plot, alpha=alpha, s=s, 
                          edgecolor='none', color=color1, marker=marker1, 
                          label=model1_name, zorder=4)
                
                all_dft_vals.append(dft_vals1)
                all_lammps_vals.append(lammps_vals1)
                
                # 计算模型1统计量（使用全部数据计算统计量）
                rmse1 = np.sqrt(np.mean((lammps_vals1 - dft_vals1)**2))
                mae1 = np.mean(np.abs(lammps_vals1 - dft_vals1))
                r2_1 = calculate_r2(dft_vals1, lammps_vals1)
            
            # 绘制模型2
            if len(dft_data2[data_type]) > 0:
                dft_vals2 = dft_data2[data_type]
                lammps_vals2 = lammps_data2[data_type]
                
                # 对Force和Virial数据进行稀疏化采样
                if data_type in ['forces', 'virial'] and len(dft_vals2) > 0:
                    n_samples = max(1, int(len(dft_vals2) * sparse_ratio))
                    indices = np.random.choice(len(dft_vals2), n_samples, replace=False)
                    dft_vals2_plot = dft_vals2[indices]
                    lammps_vals2_plot = lammps_vals2[indices]
                else:
                    dft_vals2_plot = dft_vals2
                    lammps_vals2_plot = lammps_vals2
                
                ax.scatter(dft_vals2_plot, lammps_vals2_plot, alpha=alpha, s=s,
                          edgecolor='none',color=color2, marker=marker1,
                          label=model2_name, zorder=2)
                
                all_dft_vals.append(dft_vals2)
                all_lammps_vals.append(lammps_vals2)
                
                # 计算模型2统计量（使用全部数据计算误差）
                rmse2 = np.sqrt(np.mean((lammps_vals2 - dft_vals2)**2))
                mae2 = np.mean(np.abs(lammps_vals2 - dft_vals2))
                r2_2 = calculate_r2(dft_vals2, lammps_vals2)
            
            # 绘制完美预测线
            if all_dft_vals:
                all_dft = np.concatenate(all_dft_vals)
                all_lammps = np.concatenate(all_lammps_vals)
                min_val = min(all_dft.min(), all_lammps.min())
                max_val = max(all_dft.max(), all_lammps.max())
                ax.plot([min_val, max_val], [min_val, max_val], 
                       color='gray', linewidth=1.5, linestyle='--', alpha=0.8, zorder=5)
            
            # 添加统计信息
            stats_lines = []
            if len(dft_data1[data_type]) > 0:
                stats_lines.append(f'{model1_name}:')
                stats_lines.append(f'RMSE = {rmse1:.4f}')
                stats_lines.append(f'MAE = {mae1:.4f}')
                stats_lines.append(f'R² = {r2_1:.4f}')
                stats_lines.append(f'n = {len(dft_vals1)}')
            
            if len(dft_data2[data_type]) > 0:
                if stats_lines:
                    stats_lines.append('')
                stats_lines.append(f'{model2_name}:')
                stats_lines.append(f'RMSE = {rmse2:.4f}')
                stats_lines.append(f'MAE = {mae2:.4f}')
                stats_lines.append(f'R² = {r2_2:.4f}')
                stats_lines.append(f'n = {len(dft_vals2)}')
            
            if stats_lines:
                stats_text = '\n'.join(stats_lines)
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9})
            
            # 设置标签
            ax.set_xlabel(f'DFT {data_labels[row]}', fontweight='bold')
            ax.set_ylabel(f'LAMMPS {data_labels[row]}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 为Force和Virial设置显示范围限制
            if data_type == 'forces':
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
            elif data_type == 'virial':
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
            
            # 添加图例（只在第一行显示）
            if row == 0:
                ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
            
            plot_index += 1
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="集成的双模型对比分析：自动校准并绘图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python scripts/plot_dual_model_comparison.py --nep-version 4.0.0 --dataset 2026
  
  # 指定其他数据集
  python scripts/plot_dual_model_comparison.py --nep-version 3.3.1 --dataset npj2023
  
  # 自定义稀疏化比例
  python scripts/plot_dual_model_comparison.py --nep-version 4.0.0 --dataset 2026 --sparse-ratio 0.2
        """
    )
    
    parser.add_argument(
        "--nep-version",
        type=str,
        required=True,
        help="NEP模型版本号（例如：4.0.0, 3.3.1）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称（例如：2026, npj2023, train）"
    )
    parser.add_argument(
        "--t5",
        type=float,
        default=5.0,
        help="第一列的能量阈值（默认：5.0 eV）"
    )
    parser.add_argument(
        "--t05",
        type=float,
        default=0.5,
        help="第二列的能量阈值（默认：0.5 eV）"
    )
    parser.add_argument(
        "--sparse-ratio",
        type=float,
        default=0.1,
        help="Force和Virial数据的稀疏化比例（0-1之间，默认0.1表示显示10%%的点）"
    )
    parser.add_argument(
        "--skip-alignment",
        action='store_true',
        help="跳过能量校准步骤（如果已经有校准后的文件）"
    )
    
    args = parser.parse_args()
    
    # 构建标准化的文件路径
    workspace_root = Path.cwd()
    
    nep_name = f"{args.nep_version}_{args.dataset}"
    tabgap_name = f"tabgap_{args.dataset}"
    
    nep_csv = workspace_root / "run" / "analysis" / nep_name / "energy_errors_detailed.csv"
    nep_csv_aligned = workspace_root / "run" / "analysis" / nep_name / "energy_errors_detailed_aligned.csv"
    nep_raw = workspace_root / "run" / "raw_data" / nep_name
    
    tabgap_csv = workspace_root / "run" / "analysis" / tabgap_name / "energy_errors_detailed.csv"
    tabgap_raw = workspace_root / "run" / "raw_data" / tabgap_name
    
    output_png = workspace_root / "run" / "analysis" / nep_name / "comparison_dual_2x3.png"
    
    # 检查文件是否存在
    if not nep_csv.exists():
        print(f"错误：NEP CSV文件不存在: {nep_csv}")
        return 1
    
    if not tabgap_csv.exists():
        print(f"错误：tabGAP CSV文件不存在: {tabgap_csv}")
        return 1
    
    if not nep_raw.exists():
        print(f"错误：NEP原始数据目录不存在: {nep_raw}")
        return 1
    
    if not tabgap_raw.exists():
        print(f"错误：tabGAP原始数据目录不存在: {tabgap_raw}")
        return 1
    
    print("=" * 60)
    print("双模型对比分析")
    print("=" * 60)
    print(f"NEP版本: {args.nep_version}")
    print(f"数据集: {args.dataset}")
    print(f"NEP CSV: {nep_csv}")
    print(f"NEP原始数据: {nep_raw}")
    print(f"tabGAP CSV: {tabgap_csv}")
    print(f"tabGAP原始数据: {tabgap_raw}")
    print(f"能量阈值: t5={args.t5} eV, t05={args.t05} eV")
    print(f"稀疏化比例: {args.sparse_ratio}")
    print("=" * 60)
    
    # 步骤1：能量基线校准
    if not args.skip_alignment or not nep_csv_aligned.exists():
        print("\n[步骤 1/3] 能量基线校准")
        aligned_data, success = align_nep_energies(
            str(nep_csv),
            str(nep_raw),
            str(tabgap_csv)
        )
        
        if not success:
            print("警告：能量校准验证未通过，但将继续绘图...")
        
        # 保存校准后的CSV
        with open(nep_csv_aligned, 'w', newline='') as f:
            reader_temp = csv.DictReader(open(nep_csv))
            fieldnames = reader_temp.fieldnames
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aligned_data)
        
        print(f"\n校准后的CSV已保存到: {nep_csv_aligned}")
    else:
        print(f"\n[步骤 1/3] 跳过能量校准（使用已有文件: {nep_csv_aligned}）")
    
    # 步骤2：收集数据
    print(f"\n[步骤 2/3] 收集数据")
    print(f"\n收集NEP ({args.nep_version}) 数据...")
    dft_data1, lammps_data1 = collect_data_from_csv_and_raw(str(nep_csv_aligned), str(nep_raw))
    print(f"  成功收集: 能量 {len(dft_data1['energy'])} 个, 力 {len(dft_data1['forces'])} 个, Virial {len(dft_data1['virial'])} 个")
    
    print(f"\n收集tabGAP 数据...")
    dft_data2, lammps_data2 = collect_data_from_csv_and_raw(str(tabgap_csv), str(tabgap_raw))
    print(f"  成功收集: 能量 {len(dft_data2['energy'])} 个, 力 {len(dft_data2['forces'])} 个, Virial {len(dft_data2['virial'])} 个")
    
    # 验证DFT能量基线
    if len(dft_data1['energy']) > 0 and len(dft_data2['energy']) > 0:
        mean_diff = abs(np.mean(dft_data1['energy']) - np.mean(dft_data2['energy']))
        print(f"\nDFT能量基线验证: 平均差异 = {mean_diff:.6e} eV/atom")
        if mean_diff < 1e-6:
            print("  ✓ DFT能量基线一致")
        else:
            print("  ⚠ 警告：DFT能量基线不一致")
    
    # 筛选数据
    print(f"\n筛选NEP t={args.t5} 的数据...")
    dft_data1_t5, lammps_data1_t5 = filter_data_by_threshold(dft_data1, lammps_data1, args.t5)
    print(f"  筛选后样本数: {len(dft_data1_t5['energy'])}")
    
    print(f"筛选NEP t={args.t05} 的数据...")
    dft_data1_t05, lammps_data1_t05 = filter_data_by_threshold(dft_data1, lammps_data1, args.t05)
    print(f"  筛选后样本数: {len(dft_data1_t05['energy'])}")
    
    print(f"\n筛选tabGAP t={args.t5} 的数据...")
    dft_data2_t5, lammps_data2_t5 = filter_data_by_threshold(dft_data2, lammps_data2, args.t5)
    print(f"  筛选后样本数: {len(dft_data2_t5['energy'])}")
    
    print(f"筛选tabGAP t={args.t05} 的数据...")
    dft_data2_t05, lammps_data2_t05 = filter_data_by_threshold(dft_data2, lammps_data2, args.t05)
    print(f"  筛选后样本数: {len(dft_data2_t05['energy'])}")
    
    # 步骤3：绘图
    print(f"\n[步骤 3/3] 绘制对比图")
    plot_dual_comparison_2x3(
        dft_data1_t5, lammps_data1_t5, dft_data1_t05, lammps_data1_t05,
        dft_data2_t5, lammps_data2_t5, dft_data2_t05, lammps_data2_t05,
        str(output_png),
        model1_name=f"NEP {args.nep_version}",
        model2_name="tabGAP",
        sparse_ratio=args.sparse_ratio
    )
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"校准后的CSV: {nep_csv_aligned}")
    print(f"对比图: {output_png}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

