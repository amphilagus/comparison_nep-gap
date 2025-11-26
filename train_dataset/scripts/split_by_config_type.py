#!/usr/bin/env python3
"""
将train.xyz文件按照config_type进行分割
每个config_type生成一个单独的xyz文件
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def parse_xyz_file(input_file):
    """
    解析xyz文件，按config_type分组
    
    Returns:
        dict: {config_type: [structures]}
    """
    config_groups = defaultdict(list)
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # 读取原子数
        if not lines[i].strip():
            i += 1
            continue
            
        natoms = int(lines[i].strip())
        
        # 读取注释行（包含Config_type）
        comment_line = lines[i + 1]
        
        # 提取config_type
        match = re.search(r'Config_type=(\S+)', comment_line)
        if match:
            config_type = match.group(1)
        else:
            config_type = 'unknown'
        
        # 读取完整的结构（原子数行 + 注释行 + natoms行原子坐标）
        structure_lines = lines[i:i + 2 + natoms]
        config_groups[config_type].append(''.join(structure_lines))
        
        # 移动到下一个结构
        i += 2 + natoms
    
    return config_groups


def write_split_files(config_groups, output_dir):
    """
    将分组后的结构写入各自的文件
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 写入每个config_type的文件
    for config_type, structures in config_groups.items():
        output_file = output_path / f"{config_type}.xyz"
        with open(output_file, 'w') as f:
            f.write(''.join(structures))
        
        print(f"已创建: {output_file} (包含 {len(structures)} 个结构)")


def main():
    # 设置路径
    script_dir = Path(__file__).parent
    input_file = script_dir / "train.xyz"
    output_dir = script_dir / "split_by_config_type"
    
    print(f"读取文件: {input_file}")
    
    # 解析文件
    config_groups = parse_xyz_file(input_file)
    
    print(f"\n找到 {len(config_groups)} 种config_type:")
    for config_type, structures in sorted(config_groups.items()):
        print(f"  - {config_type}: {len(structures)} 个结构")
    
    # 写入分割后的文件
    print(f"\n将文件写入目录: {output_dir}")
    write_split_files(config_groups, output_dir)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
