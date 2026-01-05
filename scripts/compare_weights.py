#!/usr/bin/env python3
"""
比较npj2023_with_weights.xyz和nep2025.xyz中相同config_type的权重是否一致
"""

import re
from collections import defaultdict


def extract_config_weights(file_path, config_key='Config_type', weight_key='Weight'):
    """从xyz文件中提取config_type和对应的权重"""
    config_weights = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # 查找包含config_type和Weight的行
        if f'{config_key}=' in line and f'{weight_key}=' in line:
            # 提取config_type (支持大小写不同的格式)
            config_match = re.search(rf'{config_key}=(\w+)', line, re.IGNORECASE)
            # 提取Weight
            weight_match = re.search(rf'{weight_key}=([\d.]+)', line)
            
            if config_match and weight_match:
                config_type = config_match.group(1)
                weight = float(weight_match.group(1))
                
                # 如果已存在该config_type，检查权重是否一致
                if config_type in config_weights:
                    if abs(config_weights[config_type] - weight) > 1e-9:
                        print(f"警告: {config_type} 在同一文件中有不同权重: {config_weights[config_type]} vs {weight}")
                else:
                    config_weights[config_type] = weight
    
    return config_weights


def compare_weights(file1, file2, file1_name, file2_name):
    """比较两个文件中的权重"""
    print(f"比较 {file1_name} 和 {file2_name} 中的权重...")
    
    # 提取权重 - 两个文件都使用Config_type (大写C)
    weights1 = extract_config_weights(file1, 'Config_type', 'Weight')
    weights2 = extract_config_weights(file2, 'Config_type', 'Weight')
    
    print(f"\n{file1_name} 中找到 {len(weights1)} 种config_type:")
    for config, weight in sorted(weights1.items()):
        print(f"  {config}: {weight}")
    
    print(f"\n{file2_name} 中找到 {len(weights2)} 种config_type:")
    for config, weight in sorted(weights2.items()):
        print(f"  {config}: {weight}")
    
    # 比较权重
    print(f"\n权重比较结果:")
    all_configs = set(weights1.keys()) | set(weights2.keys())
    
    consistent = True
    missing_in_file1 = []
    missing_in_file2 = []
    different_weights = []
    
    for config in sorted(all_configs):
        if config not in weights1:
            missing_in_file1.append(config)
            consistent = False
        elif config not in weights2:
            missing_in_file2.append(config)
            consistent = False
        else:
            weight1 = weights1[config]
            weight2 = weights2[config]
            if abs(weight1 - weight2) > 1e-9:
                different_weights.append((config, weight1, weight2))
                consistent = False
            else:
                print(f"  ✓ {config}: {weight1} (一致)")
    
    # 报告不一致的情况
    if missing_in_file1:
        print(f"\n❌ 在 {file1_name} 中缺失的config_type:")
        for config in missing_in_file1:
            print(f"  - {config}: {weights2[config]}")
    
    if missing_in_file2:
        print(f"\n❌ 在 {file2_name} 中缺失的config_type:")
        for config in missing_in_file2:
            print(f"  - {config}: {weights1[config]}")
    
    if different_weights:
        print(f"\n❌ 权重不一致的config_type:")
        for config, w1, w2 in different_weights:
            print(f"  - {config}: {file1_name}={w1}, {file2_name}={w2}")
    
    if consistent:
        print(f"\n✅ 所有相同config_type的权重都一致!")
    else:
        print(f"\n❌ 发现权重不一致的情况!")
    
    return consistent


def main():
    nep2025_file = "train_dataset/nep_baseline/nep2025.xyz"
    npj2023_file = "train_dataset/nep_baseline/npj2023.xyz"  # 直接使用原文件，它已经有权重了
    
    consistent = compare_weights(
        nep2025_file, 
        npj2023_file,
        "nep2025.xyz",
        "npj2023.xyz"
    )
    
    if consistent:
        print("\n🎉 权重添加成功，所有config_type的权重都正确匹配!")
    else:
        print("\n⚠️  发现权重不匹配的问题，请检查上述报告")


if __name__ == "__main__":
    main()