#!/usr/bin/env python3
"""
根据nep2025.xyz文件中的Config_type权重，为npj2023.xyz文件添加权重信息
"""

import re
from collections import defaultdict


def extract_weights_from_nep2025(file_path):
    """从nep2025.xyz文件中提取每个Config_type的权重"""
    config_weights = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 跳过原子数行
        if line.isdigit():
            i += 1
            continue
            
        # 查找包含Config_type和Weight的行
        if 'Config_type=' in line and 'Weight=' in line:
            # 提取Config_type
            config_match = re.search(r'Config_type=(\w+)', line)
            # 提取Weight
            weight_match = re.search(r'Weight=([\d.]+)', line)
            
            if config_match and weight_match:
                config_type = config_match.group(1)
                weight = float(weight_match.group(1))
                # 只在第一次遇到时记录和打印
                if config_type not in config_weights:
                    config_weights[config_type] = weight
                    print(f"找到权重: {config_type} = {weight}")
        
        i += 1
    
    return config_weights


def add_weights_to_npj2023(input_file, output_file, config_weights):
    """为npj2023.xyz文件添加权重信息"""
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    output_lines = []
    i = 0
    added_count = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # 如果是原子数行，直接复制
        if line.isdigit():
            output_lines.append(lines[i])
            i += 1
            continue
        
        # 如果包含config_type，添加Weight信息
        if 'config_type=' in line:
            # 提取config_type
            config_match = re.search(r'config_type=(\w+)', line)
            
            if config_match:
                config_type = config_match.group(1)
                
                # 查找对应的权重
                if config_type in config_weights:
                    weight = config_weights[config_type]
                    # 在行的开头添加Weight信息
                    modified_line = line.replace('Lattice=', f'Weight={weight} Lattice=')
                    output_lines.append(modified_line + '\n')
                    added_count += 1
                    if added_count % 100 == 0:  # 每100个打印一次进度
                        print(f"已处理 {added_count} 个构型...")
                else:
                    print(f"警告: 未找到 {config_type} 的权重信息")
                    output_lines.append(lines[i])
            else:
                output_lines.append(lines[i])
        else:
            # 其他行直接复制
            output_lines.append(lines[i])
        
        i += 1
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"总共为 {added_count} 个构型添加了权重")


def main():
    # 文件路径
    nep2025_file = "train_dataset/nep_baseline/nep2025.xyz"
    npj2023_file = "train_dataset/nep_baseline/npj2023.xyz"
    output_file = "train_dataset/nep_baseline/npj2023_with_weights.xyz"
    
    print("步骤1: 从nep2025.xyz提取权重信息...")
    config_weights = extract_weights_from_nep2025(nep2025_file)
    
    print(f"\n找到的权重映射:")
    for config_type, weight in config_weights.items():
        print(f"  {config_type}: {weight}")
    
    print(f"\n步骤2: 为npj2023.xyz添加权重...")
    add_weights_to_npj2023(npj2023_file, output_file, config_weights)
    
    print(f"\n完成! 输出文件: {output_file}")


if __name__ == "__main__":
    main()