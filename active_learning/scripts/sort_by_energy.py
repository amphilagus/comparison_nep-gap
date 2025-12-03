#!/usr/bin/env python3
"""
按能量由低到高对xyz文件中的结构进行排序

用法: python sort_by_energy.py <input.xyz> [output.xyz]
"""

import sys
import re
from pathlib import Path


def parse_structure(lines, start_idx):
    """
    从指定位置解析一个结构
    返回: (n_atoms, structure_lines, energy_per_atom)
    """
    if start_idx >= len(lines):
        return None, None, None
    
    try:
        n_atoms = int(lines[start_idx].strip())
    except (ValueError, IndexError):
        return None, None, None
    
    if start_idx + 1 >= len(lines):
        return None, None, None
    
    header_line = lines[start_idx + 1]
    
    # 提取能量（不区分大小写）
    energy_match = re.search(r'[Ee]nergy=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', header_line)
    
    if energy_match:
        total_energy = float(energy_match.group(1))
        energy_per_atom = total_energy / n_atoms
    else:
        energy_per_atom = None
    
    # 提取整个结构（原子数行 + 配置行 + 原子坐标）
    structure_lines = lines[start_idx:start_idx + 2 + n_atoms]
    
    return n_atoms, structure_lines, energy_per_atom


def read_xyz_structures(xyz_file):
    """
    读取xyz文件中的所有结构
    返回: [(energy_per_atom, structure_lines), ...]
    """
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    structures = []
    i = 0
    
    while i < len(lines):
        n_atoms, structure_lines, energy_per_atom = parse_structure(lines, i)
        
        if n_atoms is None:
            break
        
        if energy_per_atom is not None:
            structures.append((energy_per_atom, structure_lines))
        
        i += 2 + n_atoms
    
    return structures


def write_sorted_xyz(structures, output_file):
    """
    将排序后的结构写入文件
    """
    with open(output_file, 'w') as f:
        for energy, structure_lines in structures:
            for line in structure_lines:
                f.write(line)


def main():
    if len(sys.argv) < 2:
        print("用法: python sort_by_energy.py <input.xyz> [output.xyz]")
        print("示例: python sort_by_energy.py train.xyz train_sorted.xyz")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 生成输出文件名
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_sorted{input_path.suffix}")
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 读取结构
    print("\n正在读取结构...")
    structures = read_xyz_structures(input_file)
    
    if not structures:
        print("错误：未找到有效结构")
        sys.exit(1)
    
    print(f"读取到 {len(structures)} 个结构")
    
    # 按能量排序（从低到高）
    print("\n正在按能量排序...")
    structures.sort(key=lambda x: x[0])
    
    # 显示能量范围
    min_energy = structures[0][0]
    max_energy = structures[-1][0]
    print(f"能量范围: {min_energy:.6f} ~ {max_energy:.6f} eV/atom")
    
    # 写入排序后的文件
    print(f"\n正在写入排序后的文件...")
    write_sorted_xyz(structures, output_file)
    
    print(f"✓ 完成！已保存到: {output_file}")


if __name__ == '__main__':
    main()
