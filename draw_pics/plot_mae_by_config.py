#!/usr/bin/env python3
"""
MAE Error Analysis by Config Type (Dual Model Comparison)

This script analyzes LAMMPS results and plots MAE errors (Energy, Force, Virial)
grouped by config_type. Supports single or dual model comparison with grouped bars.

Key Features:
- MAE calculation for Energy, Force, and Virial
- Grouped by config_type from training set
- Dual model comparison with grouped bars
- Horizontal bar chart visualization

Usage:
    uv run python scripts/plot_mae_by_config.py -f <forcefield> [-f2 <forcefield2>] -t <train_xyz> [options]

Examples:
    # Single model: NEP potential with train dataset
    uv run python scripts/plot_mae_by_config.py -f forcefield/nep/3.3.0.txt -t train_dataset/nep_baseline/train.xyz
    
    # Dual model comparison: NEP vs TabGAP
    uv run python scripts/plot_mae_by_config.py \
      -f forcefield/nep/4.0.0.txt \
      -f2 forcefield/tabgap \
      -t train_dataset/nep_baseline/test.xyz
    
    # Custom names for dual model
    uv run python scripts/plot_mae_by_config.py \
      -f forcefield/nep/4.0.0.txt -n 4.0.0_npj2023 \
      -f2 forcefield/tabgap -n2 tabgap_npj2023 \
      -t train_dataset/nep_baseline/npj2023.xyz
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional


def parse_xyz_properties(xyz_file: str) -> Dict:
    """Parse xyz file to extract DFT reference values"""
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        raise ValueError(f"File {xyz_file} has incorrect format")
    
    n_atoms = int(lines[0].strip())
    header_line = lines[1].strip()
    
    # Extract config_type (allow letters, numbers, underscore, hyphen)
    config_type_match = re.search(r'[Cc]onfig_type=([\w\-]+)', header_line)
    config_type = config_type_match.group(1) if config_type_match else "unknown"
    
    # Extract energy (eV) - case insensitive
    energy_match = re.search(r'[Ee]nergy=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', header_line)
    energy = float(energy_match.group(1)) if energy_match else None
    energy_per_atom = energy / n_atoms if energy else None
    
    # Extract virial (eV)
    virial_match = re.search(r'virial="([^"]*)"', header_line, re.IGNORECASE)
    if virial_match:
        virial_str = virial_match.group(1)
        virial_values = [float(x) for x in virial_str.split()]
        
        if len(virial_values) == 9:
            virial = [
                virial_values[0],  # xx
                virial_values[4],  # yy
                virial_values[8],  # zz
                virial_values[1],  # xy
                virial_values[2],  # xz
                virial_values[5],  # yz
            ]
            virial_per_atom = [v / n_atoms for v in virial]
        else:
            virial_per_atom = None
    else:
        virial_per_atom = None
    
    # Extract forces (eV/Å)
    forces = []
    for i in range(2, 2 + n_atoms):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        if len(parts) >= 7:
            fx, fy, fz = float(parts[4]), float(parts[5]), float(parts[6])
            forces.append([fx, fy, fz])
    
    forces = np.array(forces) if forces else None
    
    return {
        'config_type': config_type,
        'n_atoms': n_atoms,
        'energy_per_atom': energy_per_atom,
        'virial_per_atom': virial_per_atom,
        'forces': forces,
    }


def parse_lammps_forces(dump_file: str) -> np.ndarray:
    """Parse forces from dump.forces file"""
    data = []
    with open(dump_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('ITEM:'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                fx, fy, fz = float(parts[5]), float(parts[6]), float(parts[7])
                data.append([fx, fy, fz])
    return np.array(data)


def parse_lammps_summary(summary_file: str) -> Dict:
    """Parse LAMMPS summary.txt file"""
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
        
        result = {}
        
        # Extract energy per atom (eV/atom)
        energy_patterns = [
            r'Average potential energy \(eV/atom\):\s*([-\d.eE+-]+)',
            r'Average potential energy per atom \(eV/atom\):\s*([-\d.eE+-]+)',
        ]
        
        result['energy_per_atom'] = None
        for pattern in energy_patterns:
            energy_match = re.search(pattern, content)
            if energy_match:
                result['energy_per_atom'] = float(energy_match.group(1))
                break
        
        # Extract virial components (eV/atom)
        virial_pattern = r'Average virial per atom \(eV/atom\):\s*vxx\s*=\s*([-\d.eE+-]+)\s*vyy\s*=\s*([-\d.eE+-]+)\s*vzz\s*=\s*([-\d.eE+-]+)\s*vxy\s*=\s*([-\d.eE+-]+)\s*vxz\s*=\s*([-\d.eE+-]+)\s*vyz\s*=\s*([-\d.eE+-]+)'
        virial_match = re.search(virial_pattern, content, re.DOTALL)
        
        if virial_match:
            vxx = float(virial_match.group(1))
            vyy = float(virial_match.group(2))
            vzz = float(virial_match.group(3))
            vxy = float(virial_match.group(4))
            vxz = float(virial_match.group(5))
            vyz = float(virial_match.group(6))
            result['virial_per_atom'] = [vxx, vyy, vzz, vxy, vxz, vyz]
        else:
            result['virial_per_atom'] = None
        
        return result
    except Exception:
        return {'energy_per_atom': None, 'virial_per_atom': None}


def build_config_type_map(raw_data_dir: Path) -> Dict[str, str]:
    """Build a mapping from structure_id to config_type"""
    config_map = {}
    subdirs = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()])
    
    for subdir in subdirs:
        structure_id = subdir.name
        structure_xyz = subdir / "structure.xyz"
        
        if structure_xyz.exists():
            try:
                dft_props = parse_xyz_properties(str(structure_xyz))
                config_map[structure_id] = dft_props['config_type']
            except Exception:
                pass
    
    return config_map


def collect_data_by_config(raw_data_dir: Path, config_type_map: Optional[Dict[str, str]] = None) -> Dict:
    """
    Collect data grouped by config_type
    
    Args:
        raw_data_dir: Raw data directory path
        config_type_map: Optional dict mapping structure_id to config_type (for dual model mode)
    
    Returns:
        Dict[config_type, Dict[property, List]]
    """
    print(f"\nCollecting data from: {raw_data_dir}")
    
    # Get all subdirectories
    subdirs = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()])
    
    data_by_config = defaultdict(lambda: {
        'energy_dft': [],
        'energy_lammps': [],
        'forces_dft': [],
        'forces_lammps': [],
        'virial_dft': [],
        'virial_lammps': []
    })
    
    success_count = 0
    fail_count = 0
    
    for subdir in subdirs:
        structure_id = subdir.name
        structure_xyz = subdir / "structure.xyz"
        dump_forces = subdir / "dump.forces"
        summary_txt = subdir / "summary.txt"
        
        # Check if all required files exist
        if not structure_xyz.exists() or not dump_forces.exists() or not summary_txt.exists():
            fail_count += 1
            continue
        
        try:
            # Parse DFT reference
            dft_props = parse_xyz_properties(str(structure_xyz))
            
            # Use provided config_type_map if available (for dual model mode)
            if config_type_map is not None and structure_id in config_type_map:
                config_type = config_type_map[structure_id]
            else:
                config_type = dft_props['config_type']
            
            # Parse LAMMPS predictions
            lammps_forces = parse_lammps_forces(str(dump_forces))
            lammps_summary = parse_lammps_summary(str(summary_txt))
            
            # Store energy data
            if dft_props['energy_per_atom'] is not None and lammps_summary['energy_per_atom'] is not None:
                data_by_config[config_type]['energy_dft'].append(dft_props['energy_per_atom'])
                data_by_config[config_type]['energy_lammps'].append(lammps_summary['energy_per_atom'])
            
            # Store force data
            if dft_props['forces'] is not None and len(lammps_forces) > 0:
                dft_forces_flat = dft_props['forces'].flatten()
                lammps_forces_flat = lammps_forces.flatten()
                data_by_config[config_type]['forces_dft'].extend(dft_forces_flat)
                data_by_config[config_type]['forces_lammps'].extend(lammps_forces_flat)
            
            # Store virial data
            if dft_props['virial_per_atom'] is not None and lammps_summary['virial_per_atom'] is not None:
                data_by_config[config_type]['virial_dft'].extend(dft_props['virial_per_atom'])
                data_by_config[config_type]['virial_lammps'].extend(lammps_summary['virial_per_atom'])
            
            success_count += 1
            
        except Exception as e:
            print(f"  Warning: {subdir.name} - {str(e)}")
            fail_count += 1
            continue
    
    print(f"  Data collection: {success_count} successful, {fail_count} failed")
    
    # Convert lists to numpy arrays
    for config_type in data_by_config:
        for key in data_by_config[config_type]:
            data_by_config[config_type][key] = np.array(data_by_config[config_type][key])
    
    return dict(data_by_config)


def calculate_mae_by_config(data_by_config: Dict) -> Dict[str, Dict[str, float]]:
    """
    Calculate MAE for each config_type
    
    Returns:
        Dict[config_type, Dict[property, mae_value]]
    """
    mae_results = {}
    
    for config_type, data in data_by_config.items():
        mae_results[config_type] = {}
        
        # Energy MAE (eV/atom)
        if len(data['energy_dft']) > 0:
            mae_results[config_type]['energy'] = np.mean(
                np.abs(data['energy_lammps'] - data['energy_dft'])
            )
        else:
            mae_results[config_type]['energy'] = np.nan
        
        # Force MAE (eV/Å)
        if len(data['forces_dft']) > 0:
            mae_results[config_type]['force'] = np.mean(
                np.abs(data['forces_lammps'] - data['forces_dft'])
            )
        else:
            mae_results[config_type]['force'] = np.nan
        
        # Virial MAE (eV/atom)
        if len(data['virial_dft']) > 0:
            mae_results[config_type]['virial'] = np.mean(
                np.abs(data['virial_lammps'] - data['virial_dft'])
            )
        else:
            mae_results[config_type]['virial'] = np.nan
    
    return mae_results


def plot_mae_comparison(mae_results1: Dict[str, Dict[str, float]], 
                        mae_results2: Optional[Dict[str, Dict[str, float]]] = None,
                        output_file: str = None, 
                        model1_name: str = "Model 1",
                        model2_name: str = "Model 2",
                        cutoff_multiplier: float = 4):
    """
    Plot MAE results as horizontal grouped bar chart (single or dual model)
    
    Args:
        mae_results1: Dict[config_type, Dict[property, mae_value]] for model 1
        mae_results2: Dict[config_type, Dict[property, mae_value]] for model 2 (optional)
        output_file: Output file path
        model1_name: Name of model 1
        model2_name: Name of model 2
        cutoff_multiplier: Cutoff = median * cutoff_multiplier (default: 2.5)
    """
    # Determine if dual model comparison
    dual_mode = mae_results2 is not None
    
    # Prepare data - use union of config types from both models
    if dual_mode:
        config_types = sorted(set(mae_results1.keys()) | set(mae_results2.keys()))
    else:
        config_types = sorted(mae_results1.keys())
    
    properties = ['energy', 'force', 'virial']
    property_labels = ['Energy (meV/atom)', 'Force (meV/Å)', 'Virial (meV/atom)']
    
    # Conversion factor from eV to meV
    eV_to_meV = 1000.0
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(config_types) * 0.5)))
    
    if len(config_types) == 1:
        axes = [axes]
    
    # Colors for different models
    if dual_mode:
        colors1 = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green for model 1
        colors2 = ['#85c1e9', '#f1948a', '#82e0aa']  # Lighter versions for model 2
    else:
        colors1 = ['#3498db', '#e74c3c', '#2ecc71']
        colors2 = None
    
    # Bar width and positions
    bar_height = 0.35 if dual_mode else 0.7
    y_pos = np.arange(len(config_types))
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for idx, (prop, label) in enumerate(zip(properties, property_labels)):
        ax = axes[idx]
        
        # 添加子图标题
        ax.text(-0.15, 1.02, subplot_labels[idx], transform=ax.transAxes,
                fontsize=22, fontweight='bold', ha='left', va='top')
        # Collect MAE values for model 1 (convert to meV)
        mae_values1 = []
        for config_type in config_types:
            mae_val = mae_results1.get(config_type, {}).get(prop, np.nan)
            mae_val_meV = (mae_val * eV_to_meV) if not np.isnan(mae_val) else 0.0
            mae_values1.append(mae_val_meV)
        
        # Collect MAE values for model 2 (if dual mode, convert to meV)
        if dual_mode:
            mae_values2 = []
            for config_type in config_types:
                mae_val = mae_results2.get(config_type, {}).get(prop, np.nan)
                mae_val_meV = (mae_val * eV_to_meV) if not np.isnan(mae_val) else 0.0
                mae_values2.append(mae_val_meV)
        else:
            mae_values2 = []
        
        # Calculate cutoff threshold based on median
        all_values = mae_values1 + (mae_values2 if dual_mode else [])
        valid_values = [v for v in all_values if v > 0]
        if valid_values:
            median_val = np.median(valid_values)
            cutoff = median_val * cutoff_multiplier
        else:
            cutoff = None
        
        # Prepare plotting values (original and clipped)
        mae_values1_orig = mae_values1.copy()
        mae_values1_clipped = [min(v, cutoff) if cutoff and v > 0 else v for v in mae_values1]
        is_clipped1 = [cutoff and v > cutoff for v in mae_values1]
        
        if dual_mode:
            mae_values2_orig = mae_values2.copy()
            mae_values2_clipped = [min(v, cutoff) if cutoff and v > 0 else v for v in mae_values2]
            is_clipped2 = [cutoff and v > cutoff for v in mae_values2]
        
        # Create bars for model 1
        if dual_mode:
            bars1 = ax.barh(y_pos + bar_height/2, mae_values1_clipped, bar_height, 
                           color=colors1[idx], alpha=0.8, edgecolor='black', 
                           linewidth=0.5, label=model1_name)
        else:
            bars1 = ax.barh(y_pos, mae_values1_clipped, bar_height, 
                           color=colors1[idx], alpha=0.7, edgecolor='black', 
                           linewidth=0.5)
        
        # Add value labels and cutoff marks for model 1
        for i, (bar, val_orig, val_clip, clipped) in enumerate(zip(bars1, mae_values1_orig, mae_values1_clipped, is_clipped1)):
            if val_orig > 0:
                y_center = bar.get_y() + bar.get_height()/2
                if clipped:
                    # Add zigzag cutoff marker
                    x_end = val_clip
                    zigzag_width = cutoff * 0.03
                    zigzag_height = bar.get_height() * 0.4
                    ax.plot([x_end - zigzag_width, x_end, x_end - zigzag_width], 
                           [y_center - zigzag_height, y_center, y_center + zigzag_height],
                           color='black', linewidth=2, zorder=10)
                    # Show original value
                    ax.text(val_clip - zigzag_width*1.5, y_center, 
                           f'{val_orig:.1f}', 
                           va='center', ha='right', fontsize=7, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                else:
                    ax.text(val_orig, y_center, 
                           f'{val_orig:.1f}', 
                           va='center', ha='left', fontsize=7, fontweight='bold')
        
        # Create bars and labels for model 2 (if dual mode)
        if dual_mode:
            bars2 = ax.barh(y_pos - bar_height/2, mae_values2_clipped, bar_height, 
                           color=colors2[idx], alpha=0.8, edgecolor='black', 
                           linewidth=0.5, label=model2_name)
            
            # Add value labels and cutoff marks for model 2
            for i, (bar, val_orig, val_clip, clipped) in enumerate(zip(bars2, mae_values2_orig, mae_values2_clipped, is_clipped2)):
                if val_orig > 0:
                    y_center = bar.get_y() + bar.get_height()/2
                    if clipped:
                        # Add zigzag cutoff marker
                        x_end = val_clip
                        zigzag_width = cutoff * 0.03
                        zigzag_height = bar.get_height() * 0.4
                        ax.plot([x_end - zigzag_width, x_end, x_end - zigzag_width], 
                               [y_center - zigzag_height, y_center, y_center + zigzag_height],
                               color='black', linewidth=2, zorder=10)
                        # Show original value
                        ax.text(val_clip - zigzag_width*1.5, y_center, 
                               f'{val_orig:.1f}', 
                               va='center', ha='right', fontsize=7, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                    else:
                        ax.text(val_orig, y_center, 
                               f'{val_orig:.1f}', 
                               va='center', ha='left', fontsize=7, fontweight='bold')
        
        # Customize
        ax.set_yticks(y_pos)
        
        # Only show y-axis labels on the first subplot
        # if idx == 0:

        mapping_table = {'bulk_beta_phase': r'$\beta$ phase',
                        'bulk_gamma_phase': r'$\gamma$ phase',
                        'bulk_alpha_phase': r'$\alpha$ phase',
                        'bulk_delta_phase': r'$\delta$ phase',
                        'bulk_epsilon_phase': r'$\epsilon$ phase',
                        'bulk_kappa_phase': r'$\kappa$ phase',
                        'bulk_bixbyite_phase': r'$\text{hex}^{*}$ phase',
                        'bulk_Pmc21_phase': r'$Pmc2_{1}$ phase',
                        'bulk_P-1_phase': r'$P\overline{1}$ phase',
                        'non_stoichiometry_GaO': r'GaO',
                        'non_stoichiometry_GaO2': r'GaO$_2$',
                        'non_stoichiometry_GaO3': r'GaO$_3$',
                        'non_stoichiometry_Ga3O5': r'Ga$_3$O$_5$',
                        'non_stoichiometry_Ga4O5': r'Ga$_4$O$_5$',
                        'twobody': r'dimer Ga-Ga/Ga-O/O-O',
                        'Ga_bulk': r'pure Ga',
                        'Otrimer': r'trimer O$_3$',
                        'RSS': r'random structure search',
                        'active_training': r'O clusters',
                        'melted_phase': r'melted phase',
                        'isolated_atom': r'isolated Ga/O atoms',
                        'close_3b_phase': r'close-3b phase',
                        'amorphous_phase': r'amorphous phase',
                        }

        plot_types = config_types.copy()
        for i in range(len(plot_types)):
            if plot_types[i] in mapping_table:
                plot_types[i] = mapping_table[plot_types[i]]
        ax.set_yticklabels(plot_types, fontsize=10)
        # else:
        #     ax.set_yticklabels([])  # Hide y-axis labels for right two subplots
        
        ax.set_xlabel(f'MAE: {label}', fontsize=12, fontweight='bold')
        
        # Set title
        # if dual_mode:
        #     ax.set_title(f'{label}', fontsize=14, fontweight='bold')
        # else:
        #     ax.set_title(f'{model1_name}\n{label}', fontsize=14, fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Set x-axis limits with cutoff
        if cutoff:
            ax.set_xlim(left=0, right=cutoff * 1.05)
        else:
            ax.set_xlim(left=0)
        
        # Add legend for dual mode
        # if dual_mode and idx == 0:
        ax.legend(loc='center right', fontsize=12, framealpha=0.9)
        
        # Add statistics text
        valid_mae_values1 = [v for v in mae_values1 if v > 0]
        if valid_mae_values1:
            if dual_mode:
                valid_mae_values2 = [v for v in mae_values2 if v > 0]
                mean_mae1 = np.mean(valid_mae_values1)
                max_mae1 = np.max(valid_mae_values1)
                mean_mae2 = np.mean(valid_mae_values2) if valid_mae_values2 else 0
                max_mae2 = np.max(valid_mae_values2) if valid_mae_values2 else 0
                stats_text = f'{model1_name}:\nMean: {mean_mae1:.4f}\nMax: {max_mae1:.4f}\n\n{model2_name}:\nMean: {mean_mae2:.4f}\nMax: {max_mae2:.4f}'
            else:
                mean_mae1 = np.mean(valid_mae_values1)
                max_mae1 = np.max(valid_mae_values1)
                stats_text = f'Mean: {mean_mae1:.4f}\nMax: {max_mae1:.4f}'
            
            # ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            #        verticalalignment='bottom', horizontalalignment='right',
            #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            #        fontsize=8)
    
    # Set title
    # if dual_mode:
    #     plt.suptitle(f'MAE Analysis by Config Type: {model1_name} vs {model2_name}', 
    #                 fontsize=16, fontweight='bold', y=0.98)
    # else:
    #     plt.suptitle(f'MAE Analysis by Config Type - {model1_name}', 
    #                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Print summary tables (in meV)
    eV_to_meV = 1000.0
    
    print("\n" + "=" * 100)
    print(f"MAE Summary Table - {model1_name}")
    print("=" * 100)
    print(f"{'Config Type':<30} {'Energy (meV/atom)':<20} {'Force (meV/Å)':<20} {'Virial (meV/atom)':<20}")
    print("=" * 100)
    
    for config_type in config_types:
        energy_mae = mae_results1.get(config_type, {}).get('energy', np.nan)
        force_mae = mae_results1.get(config_type, {}).get('force', np.nan)
        virial_mae = mae_results1.get(config_type, {}).get('virial', np.nan)
        
        energy_str = f"{energy_mae * eV_to_meV:.3f}" if not np.isnan(energy_mae) else "N/A"
        force_str = f"{force_mae * eV_to_meV:.3f}" if not np.isnan(force_mae) else "N/A"
        virial_str = f"{virial_mae * eV_to_meV:.3f}" if not np.isnan(virial_mae) else "N/A"
        
        print(f"{config_type:<30} {energy_str:<20} {force_str:<20} {virial_str:<20}")
    
    print("=" * 100)
    
    # Print model 2 summary if dual mode
    if dual_mode:
        print("\n" + "=" * 100)
        print(f"MAE Summary Table - {model2_name}")
        print("=" * 100)
        print(f"{'Config Type':<30} {'Energy (meV/atom)':<20} {'Force (meV/Å)':<20} {'Virial (meV/atom)':<20}")
        print("=" * 100)
        
        for config_type in config_types:
            energy_mae = mae_results2.get(config_type, {}).get('energy', np.nan)
            force_mae = mae_results2.get(config_type, {}).get('force', np.nan)
            virial_mae = mae_results2.get(config_type, {}).get('virial', np.nan)
            
            energy_str = f"{energy_mae * eV_to_meV:.3f}" if not np.isnan(energy_mae) else "N/A"
            force_str = f"{force_mae * eV_to_meV:.3f}" if not np.isnan(force_mae) else "N/A"
            virial_str = f"{virial_mae * eV_to_meV:.3f}" if not np.isnan(virial_mae) else "N/A"
            
            print(f"{config_type:<30} {energy_str:<20} {force_str:<20} {virial_str:<20}")
        
        print("=" * 100)


def generate_test_name(forcefield: str, train_xyz: str) -> str:
    """Generate test name based on forcefield and training set"""
    # Extract forcefield name
    ff_path = Path(forcefield)
    if ff_path.is_file() and ff_path.suffix == '.txt':
        ff_name = ff_path.stem
    elif ff_path.is_dir() or ff_path.name == 'tabgap':
        ff_name = 'tabgap'
    else:
        ff_name = ff_path.stem if ff_path.is_file() else ff_path.name
    
    # Extract dataset name
    train_path = Path(train_xyz)
    dataset_name = train_path.stem
    
    return f"{ff_name}_{dataset_name}"


def main():
    parser = argparse.ArgumentParser(
        description="MAE Error Analysis by Config Type (Dual Model Comparison)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model: NEP potential with train dataset
  uv run python scripts/plot_mae_by_config.py -f forcefield/nep/3.3.0.txt -t train_dataset/nep_baseline/train.xyz
  
  # Dual model: NEP vs TabGAP with test dataset
  uv run python scripts/plot_mae_by_config.py \
    -f forcefield/nep/4.0.0.txt \
    -f2 forcefield/tabgap \
    -t train_dataset/nep_baseline/test.xyz
  
  # Dual model with custom names
  uv run python scripts/plot_mae_by_config.py \
    -f forcefield/nep/4.0.0.txt -n 4.0.0_npj2023 \
    -f2 forcefield/tabgap -n2 tabgap_npj2023 \
    -t train_dataset/nep_baseline/npj2023.xyz
        """
    )
    
    parser.add_argument(
        "-f", "--forcefield",
        type=str,
        required=True,
        help="Forcefield file or directory for model 1 (.txt=NEP, dir=tabGAP)"
    )
    parser.add_argument(
        "-f2", "--forcefield2",
        type=str,
        default=None,
        help="Forcefield file or directory for model 2 (optional, for comparison)"
    )
    parser.add_argument(
        "-t", "--train-xyz",
        type=str,
        required=True,
        help="Training xyz file path"
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default=None,
        help="Test name for model 1 (optional, auto-generated if not specified)"
    )
    parser.add_argument(
        "-n2", "--name2",
        type=str,
        default=None,
        help="Test name for model 2 (optional, auto-generated if not specified)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output plot filename (optional, auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Determine if dual model mode
    dual_mode = args.forcefield2 is not None
    
    # Auto-detect potential type for model 1
    ff_path1 = Path(args.forcefield)
    if ff_path1.is_file() and ff_path1.suffix == '.txt':
        potential_type1 = "nep"
    elif ff_path1.is_dir():
        potential_type1 = "tabgap"
    else:
        print(f"Warning: Cannot identify forcefield type, defaulting to tabgap")
        potential_type1 = "tabgap"
    
    # Path management
    workspace_root = Path(__file__).parent.parent
    
    # Auto-generate test name for model 1 if not specified
    if args.name is None:
        test_name1 = generate_test_name(args.forcefield, args.train_xyz)
        print(f"Auto-generated test name (model 1): {test_name1}")
    else:
        test_name1 = args.name
    
    raw_data_dir1 = workspace_root / "run" / "raw_data" / test_name1
    
    # Setup for model 2 if dual mode
    if dual_mode:
        ff_path2 = Path(args.forcefield2)
        if ff_path2.is_file() and ff_path2.suffix == '.txt':
            potential_type2 = "nep"
        elif ff_path2.is_dir():
            potential_type2 = "tabgap"
        else:
            print(f"Warning: Cannot identify forcefield2 type, defaulting to tabgap")
            potential_type2 = "tabgap"
        
        if args.name2 is None:
            test_name2 = generate_test_name(args.forcefield2, args.train_xyz)
            print(f"Auto-generated test name (model 2): {test_name2}")
        else:
            test_name2 = args.name2
        
        raw_data_dir2 = workspace_root / "run" / "raw_data" / test_name2
        
        # Output directory - use model 1 directory
        analysis_dir = workspace_root / "run" / "analysis" / test_name1
    else:
        analysis_dir = workspace_root / "run" / "analysis" / test_name1
    
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Output plot path
    if args.output:
        output_plot = Path(args.output)
    else:
        if dual_mode:
            output_plot = analysis_dir / "mae_by_config_type_comparison.png"
        else:
            output_plot = analysis_dir / "mae_by_config_type.png"
    
    # Check if raw data directories exist
    if not raw_data_dir1.exists():
        print(f"Error: Raw data directory does not exist: {raw_data_dir1}")
        print(f"Please run the LAMMPS workflow first using run_lammps_workflow.py")
        return 1
    
    if dual_mode and not raw_data_dir2.exists():
        print(f"Error: Raw data directory for model 2 does not exist: {raw_data_dir2}")
        print(f"Please run the LAMMPS workflow first using run_lammps_workflow.py")
        return 1
    
    print("=" * 80)
    if dual_mode:
        print("MAE Error Analysis by Config Type - Dual Model Comparison")
    else:
        print("MAE Error Analysis by Config Type")
    print("=" * 80)
    print(f"\nModel 1:")
    print(f"  Test name: {test_name1}")
    print(f"  Potential type: {potential_type1.upper()}")
    print(f"  Forcefield: {args.forcefield}")
    print(f"  Raw data: {raw_data_dir1}")
    
    if dual_mode:
        print(f"\nModel 2:")
        print(f"  Test name: {test_name2}")
        print(f"  Potential type: {potential_type2.upper()}")
        print(f"  Forcefield: {args.forcefield2}")
        print(f"  Raw data: {raw_data_dir2}")
    
    print(f"\nOutput: {output_plot}")
    print("=" * 80)
    
    # Step 1: Collect data for model 1 and build config_type map
    print("\n[Step 1] Collecting data for model 1...")
    data_by_config1 = collect_data_by_config(raw_data_dir1)
    
    if not data_by_config1:
        print("Error: No data collected for model 1")
        return 1
    
    # Build config_type mapping from model 1 for dual model mode
    if dual_mode:
        print("\n  Building config_type mapping from model 1...")
        config_type_map = build_config_type_map(raw_data_dir1)
        print(f"  Mapped {len(config_type_map)} structures to config types")
    else:
        config_type_map = None
    
    print(f"\nModel 1 - Found {len(data_by_config1)} config types:")
    for config_type, data in data_by_config1.items():
        n_energy = len(data['energy_dft'])
        n_forces = len(data['forces_dft'])
        n_virial = len(data['virial_dft'])
        print(f"  {config_type}: {n_energy} energies, {n_forces} forces, {n_virial} virials")
    
    # Step 2: Calculate MAE for model 1
    print("\n[Step 2] Calculating MAE for model 1...")
    mae_results1 = calculate_mae_by_config(data_by_config1)
    
    # Steps 3-4: Collect and calculate for model 2 if dual mode
    if dual_mode:
        print("\n[Step 3] Collecting data for model 2 (using model 1's config_type mapping)...")
        data_by_config2 = collect_data_by_config(raw_data_dir2, config_type_map)
        
        if not data_by_config2:
            print("Error: No data collected for model 2")
            return 1
        
        print(f"\nModel 2 - Found {len(data_by_config2)} config types:")
        for config_type, data in data_by_config2.items():
            n_energy = len(data['energy_dft'])
            n_forces = len(data['forces_dft'])
            n_virial = len(data['virial_dft'])
            print(f"  {config_type}: {n_energy} energies, {n_forces} forces, {n_virial} virials")
        
        print("\n[Step 4] Calculating MAE for model 2...")
        mae_results2 = calculate_mae_by_config(data_by_config2)
    else:
        mae_results2 = None
    
    # Final step: Plot results
    step_num = 5 if dual_mode else 3
    print(f"\n[Step {step_num}] Plotting MAE results...")
    
    model1_name = ff_path1.stem if ff_path1.is_file() else potential_type1.upper()
    if dual_mode:
        model2_name = ff_path2.stem if ff_path2.is_file() else potential_type2.upper()
        plot_mae_comparison(mae_results1, mae_results2, str(output_plot), model1_name, model2_name)
    else:
        plot_mae_comparison(mae_results1, None, str(output_plot), model1_name)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"Output plot: {output_plot}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
