#!/usr/bin/env python3
"""
Descriptor PCA Analysis Script

This script performs PCA analysis on NEP descriptor data and visualizes
the results as a 2D scatter plot colored by configuration type.

Usage:
    uv run python scripts/pca_descriptor_analysis.py -d <descriptor_file> -x <xyz_file> [-o <output>]

Examples:
    # Basic usage
    uv run python scripts/pca_descriptor_analysis.py \\
        -d "train_dataset/check for PCA/descriptor_1.out" \\
        -x "train_dataset/check for PCA/1.xyz"
    
    # Custom output name
    uv run python scripts/pca_descriptor_analysis.py \\
        -d descriptor.out -x structures.xyz -o my_pca_analysis.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import re
from collections import Counter


def read_descriptor_file(filename):
    """Read descriptor data from file
    
    Args:
        filename: Path to descriptor file
        
    Returns:
        numpy array of shape (n_structures, n_features)
    """
    print(f"Reading descriptor file: {filename}")
    
    descriptors = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Parse descriptor values (skip the line number if present)
            parts = line.strip().split()
            # Remove line number prefix if it exists (format: "N|values")
            if '|' in parts[0]:
                parts = parts[0].split('|')[1:] + parts[1:]
            
            # Convert to floats
            values = [float(x) for x in parts]
            descriptors.append(values)
    
    descriptors = np.array(descriptors)
    print(f"  Loaded {descriptors.shape[0]} structures with {descriptors.shape[1]} descriptor dimensions")
    
    return descriptors


def read_xyz_config_types(filename):
    """Read configuration types from xyz file
    
    Args:
        filename: Path to xyz file
        
    Returns:
        List of config_type strings
    """
    print(f"Reading configuration types from: {filename}")
    
    config_types = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    while idx < len(lines):
        try:
            # Read number of atoms
            n_atoms = int(lines[idx].strip())
            
            # Read properties line
            if idx + 1 < len(lines):
                properties_line = lines[idx + 1].strip()
                
                # Extract Config_type
                match = re.search(r'Config_type=(\S+)', properties_line)
                if match:
                    config_type = match.group(1)
                else:
                    config_type = "unknown"
                
                config_types.append(config_type)
            
            # Skip to next structure
            idx += n_atoms + 2
            
        except (ValueError, IndexError):
            break
    
    print(f"  Found {len(config_types)} structures")
    
    # Print original config type statistics
    type_counts = Counter(config_types)
    print(f"\n  Original configuration type statistics:")
    for config_type, count in sorted(type_counts.items()):
        print(f"    {config_type}: {count}")
    
    return config_types


def merge_config_types(config_types):
    """Merge config types according to rules
    
    Rules:
    - Types ending with 'GPa' -> 'newly_sampled'
    - Types starting with 'v' -> 'newly_sampled'
    
    Args:
        config_types: List of original config type strings
        
    Returns:
        List of merged config type strings
    """
    print(f"\n  Merging config types...")
    
    merged_types = []
    merge_count = 0
    
    for config_type in config_types:
        # Check if ends with 'GPa' or starts with 'v'
        if config_type.endswith('GPa') or config_type.startswith('v'):
            merged_types.append('newly_sampled')
            merge_count += 1
        else:
            merged_types.append(config_type)
    
    print(f"    Merged {merge_count} structures into 'newly_sampled' category")
    
    # Print merged config type statistics
    merged_counts = Counter(merged_types)
    print(f"\n  Merged configuration type statistics:")
    for config_type, count in sorted(merged_counts.items()):
        print(f"    {config_type}: {count}")
    
    return merged_types


def perform_pca_analysis(descriptors, n_components=2):
    """Perform PCA analysis on descriptor data
    
    Args:
        descriptors: numpy array of shape (n_structures, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        transformed_data: PCA-transformed data
        pca: Fitted PCA object
        scaler: Fitted StandardScaler object
    """
    print(f"\nPerforming PCA analysis...")
    print(f"  Original dimensions: {descriptors.shape[1]}")
    print(f"  Target dimensions: {n_components}")
    
    # Standardize the data (mean=0, std=1)
    scaler = StandardScaler()
    descriptors_scaled = scaler.fit_transform(descriptors)
    print(f"  Data standardized: mean={descriptors_scaled.mean():.6f}, std={descriptors_scaled.std():.6f}")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(descriptors_scaled)
    
    # Print explained variance
    print(f"\n  Explained variance ratio:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {var_ratio*100:.2f}%")
    print(f"    Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    return transformed_data, pca, scaler


def plot_pca_results(pca_data, config_types, output_filename, pca_obj):
    """Plot PCA results as 2D scatter plot
    
    Args:
        pca_data: PCA-transformed data (n_structures, 2)
        config_types: List of configuration type labels
        output_filename: Output file path
        pca_obj: Fitted PCA object for variance info
    """
    print(f"\nGenerating PCA visualization...")
    
    # Get unique config types
    unique_types = sorted(set(config_types))
    n_types = len(unique_types)
    
    # Define color palette
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_types, 10)))
    if n_types > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, min(n_types, 20)))
    
    # Create color map
    color_map = {config_type: colors[i % len(colors)] for i, config_type in enumerate(unique_types)}
    
    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    
    # Plot each config type
    for config_type in unique_types:
        # Get indices for this config type
        indices = [i for i, ct in enumerate(config_types) if ct == config_type]
        
        # Extract data points
        x = pca_data[indices, 0]
        y = pca_data[indices, 1]
        
        # Plot scatter
        ax.scatter(x, y, c=[color_map[config_type]], 
                  label=f'{config_type} ({len(indices)})',
                  alpha=0.7, s=50, edgecolors='white', linewidths=0.5)
    
    # Get explained variance for axis labels
    var1 = pca_obj.explained_variance_ratio_[0] * 100
    var2 = pca_obj.explained_variance_ratio_[1] * 100
    
    # Labels and title
    ax.set_xlabel(f'PC1 ({var1:.2f}% variance)', fontsize=13)
    ax.set_ylabel(f'PC2 ({var2:.2f}% variance)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"  PCA plot saved to: {output_filename}")
    
    # Print data range statistics
    print(f"\n  Data range:")
    print(f"    PC1: [{pca_data[:, 0].min():.4f}, {pca_data[:, 0].max():.4f}]")
    print(f"    PC2: [{pca_data[:, 1].min():.4f}, {pca_data[:, 1].max():.4f}]")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Descriptor PCA Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run python scripts/pca_descriptor_analysis.py \\
      -d "train_dataset/check for PCA/descriptor_1.out" \\
      -x "train_dataset/check for PCA/1.xyz"
  
  # Custom output
  uv run python scripts/pca_descriptor_analysis.py \\
      -d descriptor.out -x structures.xyz -o my_pca.png
  
  # Specify number of structures to analyze
  uv run python scripts/pca_descriptor_analysis.py \\
      -d descriptor.out -x structures.xyz -n 1000
        """
    )
    
    parser.add_argument(
        "-d", "--descriptor",
        type=str,
        required=True,
        help="Path to descriptor file (e.g., descriptor_1.out)"
    )
    parser.add_argument(
        "-x", "--xyz",
        type=str,
        required=True,
        help="Path to xyz file with Config_type information"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output plot filename (default: auto-generated)"
    )
    parser.add_argument(
        "-n", "--max-structures",
        type=int,
        default=None,
        help="Maximum number of structures to analyze (default: all)"
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Skip data standardization (not recommended)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip merging config types (keep original types)"
    )
    
    args = parser.parse_args()
    
    # Check input files
    descriptor_path = Path(args.descriptor)
    xyz_path = Path(args.xyz)
    
    if not descriptor_path.exists():
        print(f"Error: Descriptor file {args.descriptor} does not exist")
        return 1
    
    if not xyz_path.exists():
        print(f"Error: XYZ file {args.xyz} does not exist")
        return 1
    
    # Set output filename
    if args.output is None:
        output_filename = descriptor_path.parent / "pca_analysis.png"
    else:
        output_filename = Path(args.output)
    
    print("=" * 80)
    print("Descriptor PCA Analysis")
    print("=" * 80)
    print(f"Descriptor file: {descriptor_path}")
    print(f"XYZ file: {xyz_path}")
    print(f"Output: {output_filename}")
    print("=" * 80)
    
    # Read data
    descriptors = read_descriptor_file(descriptor_path)
    config_types = read_xyz_config_types(xyz_path)
    
    # Merge config types if requested
    if not args.no_merge:
        config_types = merge_config_types(config_types)
    else:
        print("\n  Config type merging disabled (using original types)")
    
    # Check data consistency
    if len(descriptors) != len(config_types):
        print(f"\nWarning: Data length mismatch!")
        print(f"  Descriptors: {len(descriptors)}")
        print(f"  Config types: {len(config_types)}")
        
        min_len = min(len(descriptors), len(config_types))
        descriptors = descriptors[:min_len]
        config_types = config_types[:min_len]
        print(f"  Using first {min_len} structures")
    
    # Limit number of structures if requested
    if args.max_structures is not None and args.max_structures < len(descriptors):
        print(f"\n  Limiting to first {args.max_structures} structures")
        descriptors = descriptors[:args.max_structures]
        config_types = config_types[:args.max_structures]
    
    # Perform PCA
    pca_data, pca_obj, scaler = perform_pca_analysis(descriptors, n_components=2)
    
    # Generate plot
    plot_pca_results(pca_data, config_types, output_filename, pca_obj)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

