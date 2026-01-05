#!/usr/bin/env python3
"""
Plot MAE by Energy Bins

This script reads energy error CSV files from multiple forcefields and plots
MAE (Mean Absolute Error) grouped by energy bins relative to the lowest energy structure.

Energy bins:
- (E_min, E_min + 0.1 eV]
- (E_min + 0.1, E_min + 0.5 eV]
- (E_min + 0.5, E_min + 3.0 eV]
- (E_min + 3.0, +∞)

Usage:
    uv run python scripts/plot_mae_by_energy_bins.py \
        -c run/analysis/tabgap_npj2023/energy_errors_detailed.csv \
        -c run/analysis/4.0.0_npj2023/energy_errors_detailed.csv \
        -c run/analysis/3.3.0_npj2023/energy_errors_detailed.csv \
        -n tabGAP -n NEP-4.0.0 -n NEP-3.3.0 \
        -o energy_mae_comparison.png
"""

import argparse
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict


def parse_xyz_for_rescale(xyz_file: str) -> Dict[str, np.ndarray]:
    """
    Parse xyz file to extract energies and config_types for rescale plot
    
    Returns:
        Dict with 'energies' (per atom) and 'config_types' arrays
    """
    energies = []
    config_types = []
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
        except (ValueError, IndexError):
            break
        
        if i + 1 >= len(lines):
            break
            
        config_line = lines[i + 1]
        
        # Extract energy (case insensitive)
        energy_match = re.search(r'[Ee]nergy=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)', config_line)
        # Extract config_type
        config_type_match = re.search(r'[Cc]onfig_[Tt]ype=([\w\-]+)', config_line)
        
        if energy_match:
            energy = float(energy_match.group(1))
            avg_energy = energy / n_atoms
            energies.append(avg_energy)
            
            config_type = config_type_match.group(1) if config_type_match else 'unknown'
            config_types.append(config_type)
        
        i += 2 + n_atoms
    
    return {
        'energies': np.array(energies),
        'config_types': config_types
    }


def read_energy_errors(csv_file: str) -> Dict[str, np.ndarray]:
    """Read energy errors CSV file and return as dict of numpy arrays"""
    data = {
        'Structure_ID': [],
        'DFT_Energy_eV_per_atom': [],
        'LAMMPS_Energy_eV_per_atom': [],
        'Absolute_Error_eV_per_atom': [],
        'Relative_Error_percent': []
    }
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['Structure_ID'].append(row['Structure_ID'])
            data['DFT_Energy_eV_per_atom'].append(float(row['DFT_Energy_eV_per_atom']))
            data['LAMMPS_Energy_eV_per_atom'].append(float(row['LAMMPS_Energy_eV_per_atom']))
            data['Absolute_Error_eV_per_atom'].append(float(row['Absolute_Error_eV_per_atom']))
            data['Relative_Error_percent'].append(float(row['Relative_Error_percent']))
    
    # Convert lists to numpy arrays
    for key in ['DFT_Energy_eV_per_atom', 'LAMMPS_Energy_eV_per_atom', 
                'Absolute_Error_eV_per_atom', 'Relative_Error_percent']:
        data[key] = np.array(data[key])
    
    return data


def calculate_energy_bins(dft_energies: np.ndarray, bin_edges: List[float]) -> Tuple[List[str], np.ndarray]:
    """
    Calculate energy bins relative to minimum energy
    
    Args:
        dft_energies: DFT energy per atom values
        bin_edges: Relative energy thresholds [0.1, 0.5, 3.0]
    
    Returns:
        bin_labels: Labels for each bin
        bin_assignments: Bin index for each structure
    """
    e_min = np.min(dft_energies)
    relative_energies = dft_energies - e_min
    
    # Define bins
    bins = [0] + bin_edges + [np.inf]
    bin_assignments = np.digitize(relative_energies, bins) - 1
    
    # Create bin labels
    bin_labels = []
    for i in range(len(bins) - 1):
        if bins[i+1] == np.inf:
            bin_labels.append(f'> {bins[i]:.1f} eV')
        else:
            bin_labels.append(f'{bins[i]:.1f}-{bins[i+1]:.1f} eV')
    
    return bin_labels, bin_assignments


def calculate_mae_by_bins(data: Dict[str, np.ndarray], bin_assignments: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Calculate MAE for each energy bin
    
    Returns:
        mae_values: MAE for each bin (in eV/atom)
    """
    mae_values = []
    
    for i in range(n_bins):
        mask = bin_assignments == i
        if np.sum(mask) > 0:
            errors = data['Absolute_Error_eV_per_atom'][mask]
            mae = np.mean(errors)
            mae_values.append(mae)
        else:
            mae_values.append(0.0)
    
    return np.array(mae_values)


def plot_combined_analysis(mae_data: List[np.ndarray], 
                          bin_labels: List[str],
                          model_names: List[str],
                          train_xyz_data: Dict[str, np.ndarray],
                          epsilon: float,
                          alpha: float,
                          output_file: str):
    """
    Plot combined analysis: rescale diagram + MAE comparison
    
    Args:
        mae_data: List of MAE arrays (one per model)
        bin_labels: Labels for energy bins
        model_names: Names of models
        train_xyz_data: Training data with energies and config_types
        epsilon: Epsilon parameter for rescale
        alpha: Alpha parameter for rescale
        output_file: Output file path
    """
    n_bins = len(bin_labels)
    n_models = len(model_names)
    
    # Convert to meV
    eV_to_meV = 1000.0
    mae_data_meV = [mae * eV_to_meV for mae in mae_data]
    
    # Create figure with 2 subplots (2 rows, 1 column)
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # ========== Subplot 1: Rescale Factor Diagram ==========
    ax1 = fig.add_subplot(gs[0])
    
    energies = train_xyz_data['energies']
    config_types = train_xyz_data['config_types']
    
    # Calculate rescale factors
    e_min = np.min(energies)
    rescale_factors = 1.0 / (1.0 + (energies/epsilon - e_min/epsilon) ** alpha)
    
    # Get unique config_types
    unique_config_types = sorted(list(set(config_types)))
    config_type_to_idx = {ct: idx for idx, ct in enumerate(unique_config_types)}
    
    # Create color map
    colors_ct = plt.cm.tab10(np.linspace(0, 1, len(unique_config_types)))
    color_map = {ct: colors_ct[i] for i, ct in enumerate(unique_config_types)}
    
    # Plot theoretical rescale curve
    min_energy = np.min(energies)
    max_energy = np.max(energies)
    energy_range = np.linspace(min_energy, max_energy, 500)
    rescale_curve = 1.0 / (1.0 + (energy_range/epsilon - min_energy/epsilon) ** alpha)
    ax1.plot(energy_range, rescale_curve, 'k-', linewidth=2, zorder=1, label='Rescale Function')
    
    # Set log scale for y-axis
    ax1.set_yscale('log')
    ax1.set_xlabel('Average Energy (eV/atom)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rescale Factor', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Create twin axis for config_type
    ax1_twin = ax1.twinx()
    # ax1_twin.set_ylabel('Config Type', fontsize=12, fontweight='bold', color='gray')
    
    # Plot config_type scatter points
    for ct in unique_config_types:
        mask = [config_types[i] == ct for i in range(len(config_types))]
        e_filtered = [energies[i] for i in range(len(energies)) if mask[i]]
        ct_idx = config_type_to_idx[ct]
        ct_indices = [ct_idx] * len(e_filtered)
        ax1_twin.scatter(e_filtered, ct_indices, c=[color_map[ct]], 
                        alpha=0.6, s=20, zorder=2, label=ct if len(unique_config_types) <= 10 else None)
    
    ax1_twin.set_yticks(range(len(unique_config_types)))
    ax1_twin.set_yticklabels(unique_config_types, fontsize=8)
    ax1_twin.tick_params(axis='y', labelcolor='gray')
    
    # Add title and subplot label
    ax1.set_title(f'Weight Rescale Function (ε={epsilon}, α={alpha})', 
                 fontsize=14, fontweight='bold')
    ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes,
            fontsize=18, fontweight='bold', ha='left', va='top')
    
    # ========== Subplot 2: MAE Bar Chart ==========
    ax2 = fig.add_subplot(gs[1])
    
    # Bar positions
    x = np.arange(n_bins)
    width = 0.25
    
    # Colors for different models
    colors_model = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot bars
    for i, (mae, name) in enumerate(zip(mae_data_meV, model_names)):
        offset = (i - n_models/2 + 0.5) * width
        bars = ax2.bar(x + offset, mae, width, label=name, 
                      color=colors_model[i % len(colors_model)], alpha=0.8, 
                      edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, mae)):
            if val > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize plot
    ax2.set_xlabel('Energy Range (relative to E_min)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE (meV/atom)', fontsize=14, fontweight='bold')
    ax2.set_title('Energy MAE Comparison by Energy Bins', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, fontsize=11)
    ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    
    # Add subplot label
    ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes,
            fontsize=18, fontweight='bold', ha='left', va='top')
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("MAE Summary Table (meV/atom)")
    print("=" * 100)
    
    # Header
    header = f"{'Energy Bin':<25}"
    for name in model_names:
        header += f" {name:<20}"
    print(header)
    print("=" * 100)
    
    # Data rows
    for i, label in enumerate(bin_labels):
        row = f"{label:<25}"
        for mae in mae_data_meV:
            row += f" {mae[i]:>19.3f}"
        print(row)
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Plot MAE by Energy Bins for Multiple Forcefields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare three forcefields with rescale plot
  uv run python scripts/plot_mae_by_energy_bins.py \\
    -c run/analysis/tabgap_npj2023/energy_errors_detailed.csv \\
    -c run/analysis/4.0.0_npj2023/energy_errors_detailed.csv \\
    -c run/analysis/3.3.0_npj2023/energy_errors_detailed.csv \\
    -n tabGAP -n NEP-4.0.0 -n NEP-3.3.0 \\
    -t train_dataset/nep_baseline/npj2023.xyz \\
    --epsilon 1.0 --alpha 2.0 \\
    -o energy_mae_comparison.png
        """
    )
    
    parser.add_argument(
        "-c", "--csv",
        type=str,
        action='append',
        required=True,
        help="CSV file with energy errors (can be specified multiple times, first must be tabGAP)"
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        action='append',
        required=True,
        help="Model name corresponding to each CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="energy_mae_comparison.png",
        help="Output plot filename (default: energy_mae_comparison.png)"
    )
    parser.add_argument(
        "--bins",
        type=float,
        nargs='+',
        default=[0.1, 0.5, 3.0],
        help="Energy bin edges in eV (default: 0.1 0.5 3.0)"
    )
    parser.add_argument(
        "-t", "--train-xyz",
        type=str,
        required=True,
        help="Training XYZ file for rescale plot"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Epsilon parameter for rescale function (default: 1.0)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Alpha parameter for rescale function (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.csv) != len(args.name):
        print("Error: Number of CSV files must match number of names")
        return 1
    
    if len(args.csv) < 1:
        print("Error: At least one CSV file is required")
        return 1
    
    print("=" * 80)
    print("MAE Analysis by Energy Bins")
    print("=" * 80)
    print(f"\nNumber of forcefields: {len(args.csv)}")
    print(f"Energy bin edges: {args.bins} eV")
    print(f"Training XYZ file: {args.train_xyz}")
    print(f"Rescale parameters: ε={args.epsilon}, α={args.alpha}")
    print(f"Output file: {args.output}")
    
    # Parse training XYZ file for rescale plot
    print("\n" + "=" * 80)
    print("Parsing training XYZ file...")
    print("=" * 80)
    
    train_xyz_path = Path(args.train_xyz)
    if not train_xyz_path.exists():
        print(f"Error: Training XYZ file not found: {args.train_xyz}")
        return 1
    
    train_xyz_data = parse_xyz_for_rescale(args.train_xyz)
    print(f"  Parsed {len(train_xyz_data['energies'])} structures")
    print(f"  Energy range: {np.min(train_xyz_data['energies']):.6f} to {np.max(train_xyz_data['energies']):.6f} eV/atom")
    print(f"  Found {len(set(train_xyz_data['config_types']))} unique config types")
    
    # Read all CSV files
    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)
    
    datasets = []
    for csv_file, name in zip(args.csv, args.name):
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_file}")
            return 1
        
        data = read_energy_errors(csv_file)
        datasets.append(data)
        n_structures = len(data['DFT_Energy_eV_per_atom'])
        print(f"  {name}: {n_structures} structures from {csv_file}")
    
    # Find global minimum energy across all datasets
    all_dft_energies = []
    for data in datasets:
        all_dft_energies.extend(data['DFT_Energy_eV_per_atom'].tolist())
    global_e_min = np.min(all_dft_energies)
    
    print(f"\nGlobal minimum DFT energy: {global_e_min:.6f} eV/atom")
    
    # Calculate bins (use first dataset as reference for bin definitions)
    bin_labels, _ = calculate_energy_bins(
        datasets[0]['DFT_Energy_eV_per_atom'],
        args.bins
    )
    n_bins = len(bin_labels)
    
    print(f"\nEnergy bins (relative to E_min = {global_e_min:.6f} eV/atom):")
    for i, label in enumerate(bin_labels):
        print(f"  Bin {i+1}: {label}")
    
    # Calculate MAE for each model in each bin
    print("\n" + "=" * 80)
    print("Calculating MAE by energy bins...")
    print("=" * 80)
    
    mae_data = []
    for data, name in zip(datasets, args.name):
        dft_energies = data['DFT_Energy_eV_per_atom']
        _, bin_assignments = calculate_energy_bins(dft_energies, args.bins)
        
        mae_values = calculate_mae_by_bins(data, bin_assignments, n_bins)
        mae_data.append(mae_values)
        
        # Print structure count per bin
        print(f"\n{name}:")
        for i, label in enumerate(bin_labels):
            count = np.sum(bin_assignments == i)
            print(f"  {label}: {count} structures, MAE = {mae_values[i]*1000:.3f} meV/atom")
    
    # Plot results
    print("\n" + "=" * 80)
    print("Generating combined plot...")
    print("=" * 80)
    
    plot_combined_analysis(mae_data, bin_labels, args.name, 
                          train_xyz_data, args.epsilon, args.alpha, 
                          args.output)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

