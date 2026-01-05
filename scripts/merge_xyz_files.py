#!/usr/bin/env python3
"""
合并多个xyz文件的脚本
用法: python merge_xyz_files.py output.xyz file1.xyz file2.xyz [file3.xyz ...]
"""

import sys
import os
from pathlib import Path


def read_xyz_structures(filepath):
    """读取xyz文件中的所有结构"""
    structures = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
            
        try:
            # 读取原子数量
            natoms = int(lines[i].strip())
            
            # 读取注释行（包含晶格参数等信息）
            comment_line = lines[i + 1].strip()
            
            # 读取原子坐标
            atom_lines = []
            for j in range(i + 2, i + 2 + natoms):
                if j < len(lines):
                    atom_lines.append(lines[j].strip())
            
            # 保存结构
            structure = {
                'natoms': natoms,
                'comment': comment_line,
                'atoms': atom_lines
            }
            structures.append(structure)
            
            # 移动到下一个结构
            i = i + 2 + natoms
            
        except (ValueError, IndexError) as e:
            print(f"警告: 在文件 {filepath} 的第 {i+1} 行解析失败: {e}")
            i += 1
    
    return structures


def write_xyz_structures(filepath, structures):
    """将结构写入xyz文件"""
    with open(filepath, 'w') as f:
        for structure in structures:
            f.write(f"{structure['natoms']}\n")
            f.write(f"{structure['comment']}\n")
            for atom_line in structure['atoms']:
                f.write(f"{atom_line}\n")


def merge_xyz_files(output_file, input_files):
    """合并多个xyz文件"""
    all_structures = []
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"错误: 文件 {input_file} 不存在")
            continue
            
        print(f"正在读取文件: {input_file}")
        structures = read_xyz_structures(input_file)
        print(f"  找到 {len(structures)} 个结构")
        all_structures.extend(structures)
    
    print(f"\n总共找到 {len(all_structures)} 个结构")
    print(f"正在写入到: {output_file}")
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    write_xyz_structures(output_file, all_structures)
    print(f"合并完成! 输出文件: {output_file}")


def main():
    if len(sys.argv) < 3:
        print("用法: python merge_xyz_files.py output.xyz file1.xyz file2.xyz [file3.xyz ...]")
        print("\n示例:")
        print("python merge_xyz_files.py merged.xyz file1.xyz file2.xyz")
        print("python merge_xyz_files.py /path/to/output.xyz /path/to/file1.xyz /path/to/file2.xyz")
        sys.exit(1)
    
    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    
    print(f"输出文件: {output_file}")
    print(f"输入文件: {input_files}")
    print("-" * 50)
    
    merge_xyz_files(output_file, input_files)


if __name__ == "__main__":
    main()