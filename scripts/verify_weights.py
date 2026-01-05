#!/usr/bin/env python3
"""
验证npj2023_with_weights.xyz文件中的权重分布
"""

import re
from collections import Counter

def analyze_weights(file_path):
    """分析文件中的权重分布"""
    config_type_counts = Counter()
    weight_distribution = Counter()
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if 'Weight=' in line and 'config_type=' in line:
            # 提取权重
            weight_match = re.search(r'Weight=([\d.]+)', line)
            # 提取config_type
            config_match = re.search(r'config_type=(\w+)', line)
            
            if weight_match and config_match:
                weight = float(weight_match.group(1))
                config_type = config_match.group(1)
                
                config_type_counts[config_type] += 1
                weight_distribution[weight] += 1
    
    return config_type_counts, weight_distribution

def main():
    file_path = "train_dataset/nep_baseline/npj2023_with_weights.xyz"
    
    print("分析权重分布...")
    config_counts, weight_dist = analyze_weights(file_path)
    
    print(f"\n各Config_type的构型数量:")
    for config_type, count in sorted(config_counts.items()):
        print(f"  {config_type}: {count}")
    
    print(f"\n权重分布:")
    for weight, count in sorted(weight_dist.items()):
        print(f"  权重 {weight}: {count} 个构型")
    
    print(f"\n总构型数: {sum(config_counts.values())}")

if __name__ == "__main__":
    main()