#!/usr/bin/env python3
"""
Computational Throughput Comparison

This script plots computational throughput (million atom steps/s) vs. number of atoms
for different molecular dynamics potential models (e.g., NEP, TabGAP).

Key Features:
- Log-scale x-axis for number of atoms
- Throughput comparison across different models
- Professional visualization matching project style

Usage:
    uv run python scripts/plot_throughput_comparison.py [options]

Examples:
    # Compare NEP and TabGAP throughput
    uv run python scripts/plot_throughput_comparison.py \\
        -d NEP 10000 245.3 100000 156.8 1000000 89.2 10000000 45.6 \\
        -d TabGAP 10000 128.5 100000 82.1 1000000 48.3 10000000 23.7 \\
        -o throughput_comparison.png
    
    # Single model throughput
    uv run python scripts/plot_throughput_comparison.py \\
        -d NEP-4.5.0 10000 250 100000 160 1000000 90 10000000 46 \\
        -o nep_throughput.png
    
    # Three models comparison with custom title
    uv run python scripts/plot_throughput_comparison.py \\
        -d NEP-4.5.0 10000 245.3 100000 156.8 1000000 89.2 10000000 45.6 \\
        -d NEP-3.3.0 10000 230.1 100000 148.5 1000000 85.3 10000000 43.2 \\
        -d TabGAP 10000 128.5 100000 82.1 1000000 48.3 10000000 23.7 \\
        --title "MD Performance Comparison on NVIDIA A100" \\
        -o throughput_3models.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict


def parse_data_input(data_args: List[str]) -> List[Dict]:
    """
    Parse input data from command line arguments
    
    Args:
        data_args: List of data strings, each starting with model name followed by (n_atoms, throughput) pairs
    
    Returns:
        List of dicts with 'name', 'n_atoms', and 'throughput' keys
    
    Example:
        ["NEP", "10000", "245.3", "100000", "156.8", ...]
    """
    datasets = []
    
    for data_str in data_args:
        parts = data_str.split()
        if len(parts) < 3 or (len(parts) - 1) % 2 != 0:
            raise ValueError(f"Invalid data format: {data_str}. Expected: name n_atoms1 throughput1 n_atoms2 throughput2 ...")
        
        model_name = parts[0]
        n_atoms = []
        throughput = []
        
        for i in range(1, len(parts), 2):
            n_atoms.append(float(parts[i]))
            throughput.append(float(parts[i+1]))
        
        datasets.append({
            'name': model_name,
            'n_atoms': np.array(n_atoms),
            'throughput': np.array(throughput)
        })
    
    return datasets


def plot_throughput_comparison(datasets: List[Dict],
                               output_file: str,
                               title: str = None,
                               xlabel: str = "Number of Atoms",
                               ylabel: str = "Throughput (million atom-steps/s)",
                               show_grid: bool = True,
                               font_scale: float = 1.0):
    """
    Plot throughput comparison across different models
    
    Args:
        datasets: List of dicts with 'name', 'n_atoms', and 'throughput' keys
        output_file: Output file path
        title: Plot title (optional)
        xlabel: X-axis label
        ylabel: Y-axis label
        show_grid: Whether to show grid
        font_scale: Font size scaling factor (default: 1.0)
    """
    n_models = len(datasets)
    
    # Font sizes (base sizes that will be scaled)
    FONTSIZE_DATA_LABEL = int(9 * font_scale)
    FONTSIZE_AXIS_LABEL = int(14 * font_scale)
    FONTSIZE_TITLE = int(16 * font_scale)
    FONTSIZE_LEGEND = int(12 * font_scale)
    FONTSIZE_TICK = int(11 * font_scale)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and markers matching project style
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot each model
    for i, data in enumerate(datasets):
        model_name = data['name']
        n_atoms = data['n_atoms']
        throughput = data['throughput']
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Plot line and markers
        ax.plot(n_atoms, throughput, 
               color=color, marker=marker, linestyle=linestyle,
               markersize=10, linewidth=2.5, 
               label=model_name, alpha=0.9)
        
        # Add value labels on data points
        for x, y in zip(n_atoms, throughput):
            ax.text(x, y + 0.5, f'{y:.1f}', 
                   ha='center', va='bottom', 
                   fontsize=FONTSIZE_DATA_LABEL, fontweight='bold',
                   color=color)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Customize axes
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    
    # Set title
    if title:
        ax.set_title(title, fontsize=FONTSIZE_TITLE, fontweight='bold', pad=20)
    
    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Legend
    ax.legend(loc='upper left', fontsize=FONTSIZE_LEGEND, framealpha=0.9, 
             edgecolor='black', fancybox=True)
    
    # Format x-axis ticks
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)
    
    # Set y-axis to start from 0
    y_max = max([max(data['throughput']) for data in datasets])
    ax.set_ylim(bottom=0, top=y_max * 1.15)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("Throughput Summary Table (million atom-steps/s)")
    print("=" * 100)
    
    # Header
    header = f"{'Model':<20}"
    # Get all unique n_atoms values
    all_n_atoms = sorted(list(set([n for data in datasets for n in data['n_atoms']])))
    for n in all_n_atoms:
        if n >= 1e6:
            header += f" {n/1e6:.1f}M atoms".ljust(15)
        elif n >= 1e3:
            header += f" {n/1e3:.1f}K atoms".ljust(15)
        else:
            header += f" {int(n)} atoms".ljust(15)
    print(header)
    print("=" * 100)
    
    # Data rows
    for data in datasets:
        row = f"{data['name']:<20}"
        n_atoms_dict = {n: t for n, t in zip(data['n_atoms'], data['throughput'])}
        for n in all_n_atoms:
            if n in n_atoms_dict:
                row += f" {n_atoms_dict[n]:>13.2f} "
            else:
                row += " " + "N/A".rjust(13) + " "
        print(row)
    
    print("=" * 100)
    
    # Calculate and print speedup ratios if multiple models
    if n_models > 1:
        print("\n" + "=" * 100)
        print("Speedup Ratios (relative to first model)")
        print("=" * 100)
        
        baseline = datasets[0]
        baseline_dict = {n: t for n, t in zip(baseline['n_atoms'], baseline['throughput'])}
        
        header = f"{'Model':<20}"
        for n in all_n_atoms:
            if n >= 1e6:
                header += f" {n/1e6:.1f}M atoms".ljust(15)
            elif n >= 1e3:
                header += f" {n/1e3:.1f}K atoms".ljust(15)
            else:
                header += f" {int(n)} atoms".ljust(15)
        print(header)
        print("=" * 100)
        
        for i, data in enumerate(datasets):
            if i == 0:
                row = f"{data['name']:<20}"
                for n in all_n_atoms:
                    row += " " + "1.00x".rjust(13) + " "
                print(row)
            else:
                row = f"{data['name']:<20}"
                n_atoms_dict = {n: t for n, t in zip(data['n_atoms'], data['throughput'])}
                for n in all_n_atoms:
                    if n in n_atoms_dict and n in baseline_dict:
                        ratio = n_atoms_dict[n] / baseline_dict[n]
                        row += f" {ratio:>12.2f}x "
                    else:
                        row += " " + "N/A".rjust(13) + " "
                print(row)
        
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Computational Throughput Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare NEP and TabGAP throughput
  uv run python scripts/plot_throughput_comparison.py \\
    -d "NEP 10000 245.3 100000 156.8 1000000 89.2 10000000 45.6" \\
    -d "TabGAP 10000 128.5 100000 82.1 1000000 48.3 10000000 23.7" \\
    -o throughput_comparison.png
  
  # Single model
  uv run python scripts/plot_throughput_comparison.py \\
    -d "NEP-4.5.0 10000 250 100000 160 1000000 90 10000000 46" \\
    -o nep_throughput.png
  
  # Three models with custom title and larger font
  uv run python scripts/plot_throughput_comparison.py \\
    -d "NEP-4.5.0 10000 245.3 100000 156.8 1000000 89.2 10000000 45.6" \\
    -d "NEP-3.3.0 10000 230.1 100000 148.5 1000000 85.3 10000000 43.2" \\
    -d "TabGAP 10000 128.5 100000 82.1 1000000 48.3 10000000 23.7" \\
    --title "MD Performance Comparison on NVIDIA A100" \\
    --font-scale 1.2 \\
    -o throughput_3models.png
        """
    )
    
    parser.add_argument(
        "-d", "--data",
        type=str,
        action='append',
        required=True,
        help='Model data: "ModelName n_atoms1 throughput1 n_atoms2 throughput2 ..." (can be specified multiple times)'
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="throughput_comparison.png",
        help="Output plot filename (default: throughput_comparison.png)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (optional)"
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="Number of Atoms",
        help="X-axis label (default: Number of Atoms)"
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Throughput (million atom-steps/s)",
        help="Y-axis label (default: Throughput (million atom-steps/s))"
    )
    parser.add_argument(
        "--no-grid",
        action='store_true',
        help="Disable grid"
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Font size scaling factor (default: 1.0). Use 1.2 for larger fonts, 0.8 for smaller fonts."
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Computational Throughput Comparison")
    print("=" * 80)
    print(f"\nNumber of models: {len(args.data)}")
    print(f"Output file: {args.output}")
    print(f"Font scale: {args.font_scale}x")
    if args.title:
        print(f"Title: {args.title}")
    
    # Parse input data
    print("\n" + "=" * 80)
    print("Parsing input data...")
    print("=" * 80)
    
    try:
        datasets = parse_data_input(args.data)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Validate and print data
    for i, data in enumerate(datasets):
        print(f"\nModel {i+1}: {data['name']}")
        print(f"  Data points: {len(data['n_atoms'])}")
        print(f"  Atom range: {data['n_atoms'].min():.0f} to {data['n_atoms'].max():.0f}")
        print(f"  Throughput range: {data['throughput'].min():.2f} to {data['throughput'].max():.2f} million atom-steps/s")
    
    # Plot results
    print("\n" + "=" * 80)
    print("Generating plot...")
    print("=" * 80)
    
    plot_throughput_comparison(
        datasets=datasets,
        output_file=args.output,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        show_grid=not args.no_grid,
        font_scale=args.font_scale
    )
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"Output plot: {args.output}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

