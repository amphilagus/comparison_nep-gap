#!/usr/bin/env python3
"""
Track Diameter vs Energy Loss Scatter Plot

This script plots track diameter (nm) vs. electronic energy loss (keV/nm),
comparing theoretical simulation data with experimental data from literature.

Key Features:
- Scatter plot with error bars
- Comparison of theoretical and experimental data
- Professional visualization matching project style

Data Format:
- Column 1: Electronic energy loss (keV/nm)
- Column 2: Track diameter (nm)
- Column 3: Standard error of diameter (nm)

Usage:
    uv run python draw_pics/plot_track_diameter_vs_energy_loss.py [options]

Examples:
    # Use default input files
    uv run python draw_pics/plot_track_diameter_vs_energy_loss.py \\
        -o track_diameter_comparison.png
    
    # Custom input files and title
    uv run python draw_pics/plot_track_diameter_vs_energy_loss.py \\
        --theoretical data/theory.xlsx \\
        --experimental1 data/exp1.xlsx \\
        --experimental2 data/exp2.xlsx \\
        --title "Track Diameter vs Energy Loss in Ga₂O₃" \\
        -o track_comparison.png
    
    # Larger font and custom labels
    uv run python draw_pics/plot_track_diameter_vs_energy_loss.py \\
        --font-scale 1.2 \\
        --xlabel "Electronic Energy Loss (keV/nm)" \\
        --ylabel "Track Diameter (nm)" \\
        -o track_diameter.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Tuple


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from Excel file
    
    Args:
        filepath: Path to Excel file
    
    Returns:
        Tuple of (energy_loss, diameter, diameter_error) arrays
    
    Expected columns:
        - Column 1: Electronic energy loss (keV/nm)
        - Column 2: Track diameter (nm)
        - Column 3: Standard error of diameter (nm)
    """
    # Read Excel file without treating first row as header
    df = pd.read_excel(filepath, header=None)
    
    # Get the first three columns
    if df.shape[1] < 2:
        raise ValueError(f"File {filepath} must have at least 2 columns")
    
    energy_loss = df.iloc[:, 0].values
    diameter = df.iloc[:, 1].values
    
    # Check if error column exists
    if df.shape[1] >= 3:
        diameter_error = df.iloc[:, 2].values
    else:
        diameter_error = np.zeros_like(diameter)
    
    # Remove rows with NaN values
    valid_mask = ~(np.isnan(energy_loss) | np.isnan(diameter))
    energy_loss = energy_loss[valid_mask]
    diameter = diameter[valid_mask]
    diameter_error = diameter_error[valid_mask]
    
    return energy_loss, diameter, diameter_error


def plot_track_diameter_comparison(
        theoretical_data: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        experimental_data1: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        experimental_data2: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        output_file: str = "track_diameter_comparison.png",
        title: str = None,
        xlabel: str = "Electronic Energy Loss (keV/nm)",
        ylabel: str = "Track Diameter (nm)",
        theoretical_label: str = "MD Simulation (This Work)",
        experimental1_label: str = "Experimental Data 1",
        experimental2_label: str = "Experimental Data 2",
        show_grid: bool = True):
    """
    Plot track diameter vs energy loss comparison using ultra-fine grid
    """
    # Professional styling
    plt.rcParams['font.family'] = 'Arial'

    figsize = 10
    fontsize = 14
    
    # Ultra-fine grid division
    n = 100
    x0 = 10 * n
    y0 = 6 * n
    
    # Define total size
    
    M = x0
    N = y0
    
    # Create figure and GridSpec
    fig = plt.figure(figsize=(figsize, N/(M/figsize)))
    gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))
    
    # Create subplot
    ax = fig.add_subplot(gs[0:y0, 0:x0])
    
    # Colors and markers matching project style
    color_theory = '#3498db'      # Blue for theoretical
    color_exp1 = '#e74c3c'        # Red for experimental 1
    color_exp2 = '#2ecc71'        # Green for experimental 2
    
    marker_theory = 'o'
    marker_exp1 = 's'
    marker_exp2 = '^'
    
    # Plot theoretical data
    if theoretical_data is not None:
        energy, diameter, error = theoretical_data
        ax.errorbar(energy, diameter, yerr=error,
                   fmt=marker_theory, color=color_theory,
                   markersize=8, linewidth=2, capsize=4,
                   label=theoretical_label, alpha=0.9,
                   elinewidth=1.5, capthick=1.5)
    
    # Plot experimental data 1
    if experimental_data1 is not None:
        energy, diameter, error = experimental_data1
        ax.errorbar(energy, diameter, yerr=error,
                   fmt=marker_exp1, color=color_exp1,
                   markersize=7, linewidth=2, capsize=4,
                   label=experimental1_label, alpha=0.9,
                   elinewidth=1.5, capthick=1.5)
    
    # Plot experimental data 2
    if experimental_data2 is not None:
        energy, diameter, error = experimental_data2
        ax.errorbar(energy, diameter, yerr=error,
                   fmt=marker_exp2, color=color_exp2,
                   markersize=7, linewidth=2, capsize=4,
                   label=experimental2_label, alpha=0.9,
                   elinewidth=1.5, capthick=1.5)
    
    # Customize axes
    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=fontsize, fontweight='bold')
    
    # Set title
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=15)
    
    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Legend
    ax.legend(loc='best', fontsize=fontsize, framealpha=0.9,
             edgecolor='black', fancybox=True)
    
    # Format ticks
    ax.tick_params(axis='both', labelsize=fontsize)
    
    # Set axes to start from 0
    ax.set_xlim(left=10)
    ax.set_ylim(bottom=0)
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("Track Diameter vs Energy Loss Summary")
    print("=" * 100)
    
    datasets = []
    if theoretical_data is not None:
        datasets.append((theoretical_label, theoretical_data))
    if experimental_data1 is not None:
        datasets.append((experimental1_label, experimental_data1))
    if experimental_data2 is not None:
        datasets.append((experimental2_label, experimental_data2))
    
    for label, (energy, diameter, error) in datasets:
        print(f"\n{label}:")
        print(f"  Data points: {len(energy)}")
        print(f"  Energy loss range: {energy.min():.2f} - {energy.max():.2f} keV/nm")
        print(f"  Diameter range: {diameter.min():.2f} - {diameter.max():.2f} nm")
        if np.any(error > 0):
            print(f"  Average error: {np.mean(error[error > 0]):.2f} nm")
        
        print(f"\n  {'Energy Loss (keV/nm)':<25} {'Diameter (nm)':<20} {'Error (nm)':<15}")
        print(f"  {'-'*25} {'-'*20} {'-'*15}")
        for e, d, err in zip(energy, diameter, error):
            print(f"  {e:<25.2f} {d:<20.2f} {err:<15.2f}")
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Track Diameter vs Energy Loss Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default input files
  uv run python draw_pics/plot_track_diameter_vs_energy_loss.py \\
    -o track_diameter_comparison.png
  
  # Custom input files
  uv run python draw_pics/plot_track_diameter_vs_energy_loss.py \\
    --theoretical data/theory.xlsx \\
    --experimental1 data/exp1.xlsx \\
    --experimental2 data/exp2.xlsx \\
    -o track_comparison.png
  
  # With custom title and larger font
  uv run python draw_pics/plot_track_diameter_vs_energy_loss.py \\
    --title "Track Diameter vs Energy Loss in Ga₂O₃" \\
    --font-scale 1.2 \\
    -o track_diameter.png
        """
    )
    
    # Input files
    default_dir = Path(__file__).parent
    parser.add_argument(
        "--theoretical",
        type=str,
        default=str(default_dir / "theoritical_raw_data.xlsx"),
        help="Path to theoretical/simulation data Excel file (default: draw_pics/theoritical_raw_data.xlsx)"
    )
    parser.add_argument(
        "--experimental1",
        type=str,
        default=str(default_dir / "experimental_raw_data_1.xlsx"),
        help="Path to experimental data 1 Excel file (default: draw_pics/experimental_raw_data_1.xlsx)"
    )
    parser.add_argument(
        "--experimental2",
        type=str,
        default=str(default_dir / "experimental_raw_data_2.xlsx"),
        help="Path to experimental data 2 Excel file (default: draw_pics/experimental_raw_data_2.xlsx)"
    )
    
    # Labels for data series
    parser.add_argument(
        "--theoretical-label",
        type=str,
        default="MD   (this work)",
        help="Label for theoretical data (default: MD Simulation (This Work))"
    )
    parser.add_argument(
        "--experimental1-label",
        type=str,
        default="TEM (Ai $\mathit{et\ al.}$) ",
        help="Label for experimental data 1 (default: Ai et al. TEM)"
    )
    parser.add_argument(
        "--experimental2-label",
        type=str,
        default="TEM (Xu $\mathit{et\ al.}$ )",
        help="Label for experimental data 2 (default: Xu et al. TEM)"
    )
    
    # Output and appearance
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="track_diameter_comparison.png",
        help="Output plot filename (default: track_diameter_comparison.png)"
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
        default="Electronic Energy Loss (keV/nm)",
        help="X-axis label (default: Electronic Energy Loss (keV/nm))"
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Track Diameter (nm)",
        help="Y-axis label (default: Track Diameter (nm))"
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
    print("Track Diameter vs Energy Loss Comparison")
    print("=" * 80)
    print(f"\nOutput file: {args.output}")
    print(f"Font scale: {args.font_scale}x")
    if args.title:
        print(f"Title: {args.title}")
    
    # Load data
    print("\n" + "=" * 80)
    print("Loading data files...")
    print("=" * 80)
    
    theoretical_data = None
    experimental_data1 = None
    experimental_data2 = None
    
    # Load theoretical data
    if Path(args.theoretical).exists():
        print(f"\nLoading theoretical data from: {args.theoretical}")
        try:
            theoretical_data = load_data(args.theoretical)
            print(f"  Loaded {len(theoretical_data[0])} data points")
        except Exception as e:
            print(f"  Warning: Failed to load theoretical data: {e}")
    else:
        print(f"\nWarning: Theoretical data file not found: {args.theoretical}")
    
    # Load experimental data 1
    if Path(args.experimental1).exists():
        print(f"\nLoading experimental data 1 from: {args.experimental1}")
        try:
            experimental_data1 = load_data(args.experimental1)
            print(f"  Loaded {len(experimental_data1[0])} data points")
        except Exception as e:
            print(f"  Warning: Failed to load experimental data 1: {e}")
    else:
        print(f"\nWarning: Experimental data 1 file not found: {args.experimental1}")
    
    # Load experimental data 2
    if Path(args.experimental2).exists():
        print(f"\nLoading experimental data 2 from: {args.experimental2}")
        try:
            experimental_data2 = load_data(args.experimental2)
            print(f"  Loaded {len(experimental_data2[0])} data points")
        except Exception as e:
            print(f"  Warning: Failed to load experimental data 2: {e}")
    else:
        print(f"\nWarning: Experimental data 2 file not found: {args.experimental2}")
    
    # Check if at least one dataset was loaded
    if all(d is None for d in [theoretical_data, experimental_data1, experimental_data2]):
        print("\nError: No valid data files found. Please check file paths.")
        return 1
    
    # Plot results
    print("\n" + "=" * 80)
    print("Generating plot with ultra-fine grid...")
    print("=" * 80)
    
    plot_track_diameter_comparison(
        theoretical_data=theoretical_data,
        experimental_data1=experimental_data1,
        experimental_data2=experimental_data2,
        output_file=args.output,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        theoretical_label=args.theoretical_label,
        experimental1_label=args.experimental1_label,
        experimental2_label=args.experimental2_label,
        show_grid=not args.no_grid
    )
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"Output plot: {args.output}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

