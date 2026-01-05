#!/usr/bin/env python3
"""
详细分析两个文件中的权重分布和统计信息
"""

import re
from collections import Counter, defaultdict


def analyze_file_weights(file_path, file_name):
    """分析单个文件的权重分布"""
    config_counts = Counter()
    weight_counts = Counter()
    config_to_weight = {}
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if 'Config_type=' in line and 'Weight=' in line:
            # 提取config_type
            config_match = re.search(r'Config_type=(\w+)', line)
            # 提取Weight
            weight_match = re.search(r'Weight=([\d.]+)', line)
            
            if config_match and weight_match:
                config_type = config_match.group(1)
                weight = float(weight_match.group(1))
                
                config_counts[config_type] += 1
                weight_counts[weight] += 1
                config_to_weight[config_type] = weight
    
    return config_counts, weight_counts, config_to_weight


def main():
    nep2025_file = "train_dataset/nep_baseline/nep2025.xyz"
    npj2023_file = "train_dataset/nep_baseline/npj2023.xyz"
    
    print("=== 详细权重分析 ===\n")
    
    # 分析nep2025.xyz
    print("📊 分析 nep2025.xyz:")
    config_counts_2025, weight_counts_2025, config_weights_2025 = analyze_file_weights(nep2025_file, "nep2025.xyz")
    
    print(f"  总构型数: {sum(config_counts_2025.values())}")
    print(f"  Config_type种类: {len(config_counts_2025)}")
    print(f"  权重种类: {len(weight_counts_2025)}")
    
    print(f"\n  各Config_type构型数量:")
    for config_type, count in sorted(config_counts_2025.items()):
        weight = config_weights_2025[config_type]
        print(f"    {config_type}: {count} 个构型 (权重: {weight})")
    
    print(f"\n  权重分布:")
    for weight, count in sorted(weight_counts_2025.items()):
        print(f"    权重 {weight}: {count} 个构型")
    
    # 分析npj2023.xyz
    print(f"\n📊 分析 npj2023.xyz:")
    config_counts_2023, weight_counts_2023, config_weights_2023 = analyze_file_weights(npj2023_file, "npj2023.xyz")
    
    print(f"  总构型数: {sum(config_counts_2023.values())}")
    print(f"  Config_type种类: {len(config_counts_2023)}")
    print(f"  权重种类: {len(weight_counts_2023)}")
    
    print(f"\n  各Config_type构型数量:")
    for config_type, count in sorted(config_counts_2023.items()):
        weight = config_weights_2023[config_type]
        print(f"    {config_type}: {count} 个构型 (权重: {weight})")
    
    print(f"\n  权重分布:")
    for weight, count in sorted(weight_counts_2023.items()):
        print(f"    权重 {weight}: {count} 个构型")
    
    # 比较分析
    print(f"\n🔍 比较分析:")
    
    # 检查权重一致性
    all_consistent = True
    for config_type in set(config_weights_2025.keys()) | set(config_weights_2023.keys()):
        if config_type in config_weights_2025 and config_type in config_weights_2023:
            if abs(config_weights_2025[config_type] - config_weights_2023[config_type]) > 1e-9:
                print(f"  ❌ {config_type}: 权重不一致 ({config_weights_2025[config_type]} vs {config_weights_2023[config_type]})")
                all_consistent = False
        elif config_type in config_weights_2025:
            print(f"  ⚠️  {config_type}: 仅在nep2025.xyz中存在")
        else:
            print(f"  ⚠️  {config_type}: 仅在npj2023.xyz中存在")
    
    if all_consistent and set(config_weights_2025.keys()) == set(config_weights_2023.keys()):
        print(f"  ✅ 所有Config_type的权重完全一致!")
    
    # 构型数量比较
    print(f"\n📈 构型数量比较:")
    all_configs = set(config_counts_2025.keys()) | set(config_counts_2023.keys())
    for config_type in sorted(all_configs):
        count_2025 = config_counts_2025.get(config_type, 0)
        count_2023 = config_counts_2023.get(config_type, 0)
        if count_2025 != count_2023:
            print(f"  📊 {config_type}: nep2025={count_2025}, npj2023={count_2023} (差异: {count_2023-count_2025})")
    
    print(f"\n📋 总结:")
    print(f"  nep2025.xyz: {sum(config_counts_2025.values())} 个构型")
    print(f"  npj2023.xyz: {sum(config_counts_2023.values())} 个构型")
    print(f"  权重映射: {'✅ 完全一致' if all_consistent else '❌ 存在差异'}")


if __name__ == "__main__":
    main()