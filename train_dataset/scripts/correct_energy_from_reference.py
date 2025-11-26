#!/usr/bin/env python3
"""
根据参考训练集修正目标训练集的能量

用法: python correct_energy_from_reference.py <reference.xyz> <target.xyz> <output.xyz> [tolerance]

匹配方法：比较Lattice和Virial数值（使用相对容差）
如果目标训练集中的结构未在参考训练集中找到，则报错
默认相对容差：1e-6
"""

import sys
import re
import numpy as np


def extract_lattice_virial(config_line):
    """从配置行提取Lattice和Virial数组"""
    lattice_match = re.search(r'Lattice="([^"]+)"', config_line)
    virial_match = re.search(r'Virial="([^"]+)"', config_line)
    
    if not lattice_match or not virial_match:
        return None, None
    
    # 转换为浮点数数组
    lattice_str = lattice_match.group(1)
    virial_str = virial_match.group(1)
    
    lattice = np.array([float(x) for x in lattice_str.split()])
    virial = np.array([float(x) for x in virial_str.split()])
    
    return lattice, virial


def structures_match(struct1, struct2, rtol=1e-6):
    """判断两个结构是否匹配（使用相对容差）"""
    # 原子数必须相同
    if struct1['n_atoms'] != struct2['n_atoms']:
        return False
    
    # 比较Lattice
    if not np.allclose(struct1['lattice'], struct2['lattice'], rtol=rtol, atol=0):
        return False
    
    # 比较Virial
    if not np.allclose(struct1['virial'], struct2['virial'], rtol=rtol, atol=0):
        return False
    
    return True


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
        
        # 提取Lattice和Virial
        lattice, virial = extract_lattice_virial(config_line)
        
        if lattice is not None and virial is not None:
            # 提取能量
            energy_match = re.search(r'[Ee]nergy=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', config_line)
            energy = float(energy_match.group(1)) if energy_match else None
            
            # 保存结构信息
            structure = {
                'n_atoms': n_atoms,
                'config_line': config_line,
                'atom_lines': lines[i + 2:i + 2 + n_atoms],
                'lattice': lattice,
                'virial': virial,
                'energy': energy,
                'line_index': i
            }
            structures.append(structure)
        
        # 移动到下一个结构
        i += 2 + n_atoms
    
    return structures


def correct_energies(reference_structures, target_structures, rtol=1e-6):
    """根据参考结构修正目标结构的能量"""
    print(f"参考训练集结构数: {len(reference_structures)}")
    print(f"目标训练集结构数: {len(target_structures)}")
    print(f"相对容差: {rtol}")
    print()
    
    # 匹配并修正能量
    matched_count = 0
    unmatched_structures = []
    
    for idx, target_struct in enumerate(target_structures):
        # 在参考结构中查找匹配
        matched = False
        
        for ref_struct in reference_structures:
            if structures_match(target_struct, ref_struct, rtol=rtol):
                # 找到匹配的结构，使用参考能量
                target_struct['corrected_energy'] = ref_struct['energy']
                matched_count += 1
                matched = True
                break
        
        if not matched:
            # 未找到匹配的结构
            unmatched_structures.append({
                'index': idx,
                'lattice': target_struct['lattice'],
                'virial': target_struct['virial'],
                'line': target_struct['line_index']
            })
    
    # 错误检测：如果有未匹配的结构，报错
    if unmatched_structures:
        print("错误: 以下目标训练集中的结构未在参考训练集中找到:")
        print()
        for unmatch in unmatched_structures[:10]:  # 只显示前10个
            print(f"  结构索引 {unmatch['index']} (文件行 {unmatch['line']}):")
            print(f"    Lattice: {unmatch['lattice']}")
            print(f"    Virial:  {unmatch['virial']}")
            print()
        
        if len(unmatched_structures) > 10:
            print(f"  ... 还有 {len(unmatched_structures) - 10} 个未匹配的结构")
        
        print(f"总计 {len(unmatched_structures)} 个结构未找到匹配")
        sys.exit(1)
    
    print(f"成功匹配: {matched_count}/{len(target_structures)} 个结构")
    return target_structures


def write_corrected_xyz(structures, output_filename):
    """写入修正能量后的xyz文件"""
    with open(output_filename, 'w') as f:
        for structure in structures:
            # 写入原子数
            f.write(f"{structure['n_atoms']}\n")
            
            # 更新配置行中的能量
            config_line = structure['config_line']
            new_config_line = re.sub(
                r'([Ee]nergy=)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)',
                lambda m: f"{m.group(1)}{structure['corrected_energy']}",
                config_line
            )
            f.write(new_config_line)
            
            # 写入原子坐标
            for atom_line in structure['atom_lines']:
                f.write(atom_line)
    
    print(f"已写入修正后的文件: {output_filename}")


def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("用法: python correct_energy_from_reference.py <reference.xyz> <target.xyz> <output.xyz> [tolerance]")
        print("示例: python correct_energy_from_reference.py train.xyz train_sparsed.xyz train_corrected.xyz")
        print("      python correct_energy_from_reference.py train.xyz train_sparsed.xyz train_corrected.xyz 1e-5")
        print()
        print("默认相对容差: 1e-6")
        sys.exit(1)
    
    reference_file = sys.argv[1]
    target_file = sys.argv[2]
    output_file = sys.argv[3]
    rtol = float(sys.argv[4]) if len(sys.argv) == 5 else 1e-6
    
    print(f"参考训练集: {reference_file}")
    print(f"目标训练集: {target_file}")
    print(f"输出文件: {output_file}")
    print(f"相对容差: {rtol}")
    print()
    
    # 解析参考训练集
    print("正在解析参考训练集...")
    reference_structures = parse_xyz_file(reference_file)
    
    # 解析目标训练集
    print("正在解析目标训练集...")
    target_structures = parse_xyz_file(target_file)
    print()
    
    # 修正能量
    print("正在匹配结构并修正能量...")
    corrected_structures = correct_energies(reference_structures, target_structures, rtol=rtol)
    print()
    
    # 写入新文件
    write_corrected_xyz(corrected_structures, output_file)
    
    # 打印统计信息
    print()
    print("能量修正完成!")
    original_energies = [s['energy'] for s in target_structures if s['energy'] is not None]
    corrected_energies = [s['corrected_energy'] for s in corrected_structures]
    
    if original_energies and corrected_energies:
        print(f"  原始能量范围: {min(original_energies):.6f} - {max(original_energies):.6f}")
        print(f"  修正能量范围: {min(corrected_energies):.6f} - {max(corrected_energies):.6f}")


if __name__ == '__main__':
    main()
