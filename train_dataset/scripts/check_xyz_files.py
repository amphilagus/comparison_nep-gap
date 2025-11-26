#!/usr/bin/env python3
"""
检查 train_dataset/split_by_config_type 目录中的 xyz 文件
判断每个文件中的所有结构是否具有相同的原子数
如果是坏文件，则按原子数分割成多个好文件
"""

from pathlib import Path
from ase.io import read, write
from collections import defaultdict


def check_xyz_file(filepath):
    """
    检查一个xyz文件中所有结构的原子数是否一致
    
    Returns:
        tuple: (is_good, atom_counts, total_structures, structures)
            - is_good: bool, 是否为好文件
            - atom_counts: set, 文件中出现的所有不同原子数
            - total_structures: int, 结构总数
            - structures: list, 所有结构的列表
    """
    try:
        structures = read(filepath, index=':')
        
        # 如果只有一个结构，也算作列表
        if not isinstance(structures, list):
            structures = [structures]
        
        atom_counts = set()
        for structure in structures:
            atom_counts.add(len(structure))
        
        is_good = len(atom_counts) == 1
        return is_good, atom_counts, len(structures), structures
    
    except Exception as e:
        print(f"  错误: 无法读取文件 - {e}")
        return None, None, 0, None


def split_bad_file(filepath, structures):
    """
    将坏文件按原子数分割成多个好文件
    
    Args:
        filepath: Path, 原文件路径
        structures: list, 所有结构的列表
    
    Returns:
        list: 生成的新文件路径列表
    """
    # 按原子数分组
    groups = defaultdict(list)
    for structure in structures:
        atom_count = len(structure)
        groups[atom_count].append(structure)
    
    # 生成新文件
    output_dir = filepath.parent
    base_name = filepath.stem  # 不带扩展名的文件名
    
    created_files = []
    for idx, (atom_count, group_structures) in enumerate(sorted(groups.items()), start=1):
        output_file = output_dir / f"{base_name}_{idx}.xyz"
        write(output_file, group_structures)
        created_files.append((output_file, atom_count, len(group_structures)))
    
    return created_files


def main():
    xyz_dir = Path("train_dataset/split_by_config_type")
    
    if not xyz_dir.exists():
        print(f"错误: 目录不存在 - {xyz_dir}")
        return
    
    xyz_files = sorted(xyz_dir.glob("*.xyz"))
    
    if not xyz_files:
        print(f"在 {xyz_dir} 中没有找到 xyz 文件")
        return
    
    good_files = []
    bad_files = []
    split_info = []
    
    print("=" * 80)
    print("检查 xyz 文件中的原子数一致性")
    print("=" * 80)
    print()
    
    for xyz_file in xyz_files:
        print(f"检查: {xyz_file.name}")
        is_good, atom_counts, total_structures, structures = check_xyz_file(xyz_file)
        
        if is_good is None:
            continue
        
        if is_good:
            good_files.append(xyz_file.name)
            print(f"  ✓ 好文件 - 所有 {total_structures} 个结构都有 {list(atom_counts)[0]} 个原子")
        else:
            bad_files.append(xyz_file.name)
            print(f"  ✗ 坏文件 - {total_structures} 个结构中原子数不一致: {sorted(atom_counts)}")
            
            # 分割坏文件
            print(f"  → 正在分割文件...")
            created_files = split_bad_file(xyz_file, structures)
            split_info.append((xyz_file.name, created_files))
            
            for new_file, atom_count, count in created_files:
                print(f"    创建: {new_file.name} ({count} 个结构, 每个 {atom_count} 个原子)")
        print()
    
    # 总结
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print(f"\n好文件 ({len(good_files)} 个):")
    for filename in good_files:
        print(f"  ✓ {filename}")
    
    print(f"\n坏文件 ({len(bad_files)} 个) - 已分割:")
    for filename in bad_files:
        print(f"  ✗ {filename}")
    
    if split_info:
        print(f"\n分割详情:")
        for original_name, created_files in split_info:
            print(f"\n  {original_name} → 分割成 {len(created_files)} 个文件:")
            for new_file, atom_count, count in created_files:
                print(f"    - {new_file.name}: {count} 个结构 (每个 {atom_count} 个原子)")
    
    print(f"\n总计: {len(xyz_files)} 个文件, {len(good_files)} 个好文件, {len(bad_files)} 个坏文件已分割")


if __name__ == "__main__":
    main()
