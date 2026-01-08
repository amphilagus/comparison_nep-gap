#!/usr/bin/env python3
"""
Computational Throughput Comparison

This script plots computational throughput (million atom steps/s) vs. number of atoms
for different molecular dynamics potential models (e.g., NEP, tabGAP).

Key Features:
- Log-scale x-axis for number of atoms
- Throughput comparison across different models
- Professional visualization matching project style

Usage:
    uv run python scripts/plot_throughput_comparison.py [options]

Examples:
    # Compare NEP and tabGAP throughput
    uv run python scripts/plot_throughput_comparison.py \\
        -d NEP 10000 245.3 100000 156.8 1000000 89.2 10000000 45.6 \\
        -d tabGAP 10000 128.5 100000 82.1 1000000 48.3 10000000 23.7 \\
        -o throughput_comparison.png
    
    # Single model throughput
    uv run python scripts/plot_throughput_comparison.py \\
        -d NEP-4.5.0 10000 250 100000 160 1000000 90 10000000 46 \\
        -o nep_throughput.png
    
    # Three models comparison with custom title
    uv run python scripts/plot_throughput_comparison.py \\
        -d NEP-4.5.0 10000 245.3 100000 156.8 1000000 89.2 10000000 45.6 \\
        -d NEP-3.3.0 10000 230.1 100000 148.5 1000000 85.3 10000000 43.2 \\
        -d tabGAP 10000 128.5 100000 82.1 1000000 48.3 10000000 23.7 \\
        --title "MD Performance Comparison on NVIDIA A100" \\
        -o throughput_3models.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd


def plot_throughput_comparison(datasets: List[Dict],
                               output_file: str,
                               title: str = None,
                               xlabel: str = "Number of Atoms",
                               ylabel: str = "Throughput (million atom-steps/s)",
                               show_grid: bool = True):
    """
    Plot throughput comparison across different models using ultra-fine grid
    """
    # Professional styling
    plt.rcParams['font.family'] = 'Arial'
    figsize = 10
    fontsize = 14
    
    # Ultra-fine grid division
    n = 100
    x0 = 10 * n
    y0 = 6 * n
    
    # Define margins and total size
    M = x0
    N = y0
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=(figsize, N/(M/figsize)))
    gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))
    
    # Create subplot
    ax = fig.add_subplot(gs[0:y0, 0:x0])
    
    # Colors and markers
    colors = {'NEP': '#e74c3c', 'tabGAP': '#3498db'}
    markers = {'NEP': 'o', 'tabGAP': 's'}
    linestyles = {'NEP': '-', 'tabGAP': '--'}
    
    # Plot each model
    for data in datasets:
        model_name = data['name']
        n_atoms = data['n_atoms']
        throughput_mean = data['throughput_mean']
        throughput_std = data['throughput_std']
        
        color = colors.get(model_name, '#2ecc71')
        marker = markers.get(model_name, '^')
        linestyle = linestyles.get(model_name, '-')
        
        # Plot line with error bars
        ax.errorbar(n_atoms, throughput_mean, yerr=throughput_std,
                   color=color, marker=marker, linestyle=linestyle,
                   markersize=8, linewidth=2, capsize=4,
                   label=model_name, alpha=0.9)
        
        # Add value labels
        for x, y in zip(n_atoms, throughput_mean):
            ax.text(x, y * 1.05, f'{y:.1f}', 
                   ha='center', va='bottom', 
                   fontsize=fontsize, fontweight='bold',
                   color=color)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Customize axes
    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold')
    
    # if title:
    #     ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.tick_params(axis='both', labelsize=fontsize)
    
    # Legend
    ax.legend(loc='upper left', fontsize=fontsize)
    
    # Set y-axis limits
    y_max = max([max(data['throughput_mean'] + data['throughput_std']) for data in datasets])
    ax.set_ylim(bottom=0, top=y_max * 1.25)
    
    # Save figure
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
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
        n_atoms_dict = {n: t for n, t in zip(data['n_atoms'], data['throughput_mean'])}
        for n in all_n_atoms:
            if n in n_atoms_dict:
                row += f" {n_atoms_dict[n]:>13.2f} "
            else:
                row += " " + "N/A".rjust(13) + " "
        print(row)
    
    print("=" * 100)
    
    # Calculate and print speedup ratios if multiple models
    if len(datasets) > 1:
        print("\n" + "=" * 100)
        print("Speedup Ratios (relative to tabGAP)")
        print("=" * 100)
        
        # Find tabGAP as baseline
        baseline = None
        for data in datasets:
            if data['name'] == 'tabGAP':
                baseline = data
                break
        
        if baseline:
            baseline_dict = {n: t for n, t in zip(baseline['n_atoms'], baseline['throughput_mean'])}
            
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
            
            for data in datasets:
                row = f"{data['name']:<20}"
                n_atoms_dict = {n: t for n, t in zip(data['n_atoms'], data['throughput_mean'])}
                for n in all_n_atoms:
                    if n in n_atoms_dict and n in baseline_dict and baseline_dict[n] > 0:
                        ratio = n_atoms_dict[n] / baseline_dict[n]
                        row += f" {ratio:>12.2f}x "
                    else:
                        row += " " + "N/A".rjust(13) + " "
                print(row)
            
            print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Computational Throughput Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="draw_pics/theoritical_raw_data.csv",
        help="Input CSV file (default: draw_pics/theoritical_raw_data.csv)"
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
        default="Computational Throughput Comparison: NEP vs tabGAP",
        help="Plot title"
    )
    
    args = parser.parse_args()
    
    workspace_root = Path(__file__).parent.parent
    input_path = workspace_root / args.input
    
    print("=" * 80)
    print("Computational Throughput Comparison")
    print("=" * 80)
    print(f"Input file: {input_path}")
    print(f"Output file: {args.output}")
    
    # Read data from CSV
    try:
        df = pd.read_csv(input_path)
        print("\nSuccessfully loaded data from CSV:")
        print(df)
        
        n_atoms = df.iloc[:, 0].values
        
        # NEP data (columns 1, 2, 3)
        nep_raw = df.iloc[:, 1:4].values
        nep_mean = np.mean(nep_raw, axis=1)
        nep_std = np.std(nep_raw, axis=1)
        
        # tabGAP data (columns 4, 5, 6)
        gap_raw = df.iloc[:, 4:7].values
        gap_mean = np.mean(gap_raw, axis=1)
        gap_std = np.std(gap_raw, axis=1)
        
        datasets = [
            {
                'name': 'NEP',
                'n_atoms': n_atoms,
                'throughput_mean': nep_mean,
                'throughput_std': nep_std
            },
            {
                'name': 'tabGAP',
                'n_atoms': n_atoms,
                'throughput_mean': gap_mean,
                'throughput_std': gap_std
            }
        ]
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 1
    
    # Plot results
    print("\n" + "=" * 80)
    print("Generating plot with ultra-fine grid...")
    print("=" * 80)
    
    plot_throughput_comparison(
        datasets=datasets,
        output_file=args.output,
        title=args.title
    )
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

