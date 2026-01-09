#!/usr/bin/env python3
"""
Comprehensive Analysis: 3 Models Energy Comparison + PCA in 2x2 Panel

Panel 1-3: Energy scatter plots (DFT vs LAMMPS) with config_type coloring
Panel 4: PCA analysis of descriptors

Features:
- Supports separate test xyz files for NEP and tabGAP models
- Automatically detects model type (NEP vs tabGAP)
- Config_type coloring for easy comparison
- PCA analysis with optional config type merging

All parameters are set via hard-coded variables near the top of the file.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

test_xyz = "train_dataset/gap_baseline/active_training/train.xyz"
tabgap_test_xyz = "train_dataset/gap_baseline/active_training/train.xyz"
models = [
    "forcefield/tabgap",
    "forcefield/nep/4.2.0.txt",
    "forcefield/nep/5.1.0.txt",
]
names = [
    r"$\bf{tabGAP}$",
    r"$\bf{NEP\,(GAP\ dataset)}$",
    r"$\bf{NEP\,(augmented\ dataset)}$",
]
descriptor_file = "train_dataset/check for PCA/descriptor_1.out"
descriptor_xyz = "train_dataset/check for PCA/1.xyz"
output_file = "draw_pics/output/melting_pca.png"
lammps_exe = "opt/lmp_nep_tabgap"
skip_run = True
merge_types = True
nep_to_tabgap_baseline = True

original_zpe = {
    "Ga": -0.0244486,
    "O": -0.0350174,
}
nep89_zpe = {
    "Ga": -1.68768,
    "O": -3.19589,
}
energy_diff = {
    "Ga": nep89_zpe["Ga"] - original_zpe["Ga"],
    "O": nep89_zpe["O"] - original_zpe["O"],
}


def parse_xyz_structure(lines, start_idx):
    """Parse a single XYZ structure"""
    if start_idx >= len(lines):
        return None, start_idx
    
    try:
        num_atoms = int(lines[start_idx].strip())
        properties_line = lines[start_idx + 1].strip()
        
        config_type_match = re.search(r'Config_type=(\S+)', properties_line)
        config_type = config_type_match.group(1) if config_type_match else "unknown"
        
        energy_match = re.search(r'Energy=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)', properties_line)
        dft_energy = float(energy_match.group(1)) if energy_match else 0.0
        
        atom_counts = {"Ga": 0, "O": 0}
        for j in range(num_atoms):
            atom_line_idx = start_idx + 2 + j
            if atom_line_idx >= len(lines):
                break
            parts = lines[atom_line_idx].split()
            if not parts:
                continue
            element = parts[0]
            if element in atom_counts:
                atom_counts[element] += 1

        return {
            'num_atoms': num_atoms,
            'config_type': config_type,
            'dft_energy': dft_energy,
            'dft_energy_per_atom': dft_energy / num_atoms,
            'atom_counts': atom_counts,
        }, start_idx + 2 + num_atoms
        
    except Exception:
        return None, start_idx + 1


def read_test_xyz_structures(filename):
    """Read test.xyz and parse all structures"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    structures = []
    idx = 0
    
    while idx < len(lines):
        structure, idx = parse_xyz_structure(lines, idx)
        if structure:
            structures.append(structure)
    
    return structures


def extract_potential_energy_from_log(log_file):
    """Extract potential energy from LAMMPS log file"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        in_thermo = False
        pe_col_index = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Step') and 'PotEng' in line:
                in_thermo = True
                headers = line.split()
                try:
                    pe_col_index = headers.index('PotEng')
                except ValueError:
                    continue
                continue
            
            if in_thermo and (line.startswith('Loop') or line.startswith('---')):
                break
            
            if in_thermo and pe_col_index is not None:
                parts = line.split()
                if len(parts) > pe_col_index:
                    try:
                        return float(parts[pe_col_index])
                    except ValueError:
                        continue
        
        return None
    except Exception:
        return None


def collect_predictions(raw_data_dir, structures, potential_type):
    """Collect DFT and LAMMPS predictions grouped by config_type"""
    raw_data_path = Path(raw_data_dir)
    subdirs = sorted([d for d in raw_data_path.iterdir() if d.is_dir()])
    
    dft_energies = []
    lammps_energies = []
    config_types = []
    
    for i, subdir in enumerate(subdirs):
        if i >= len(structures):
            break
        
        log_file = subdir / "lammps.log"
        if not log_file.exists():
            continue
        
        predicted_energy = extract_potential_energy_from_log(log_file)
        if predicted_energy is None:
            continue
        
        structure = structures[i]
        dft_energy_per_atom = structure['dft_energy_per_atom']
        if potential_type == "nep" and nep_to_tabgap_baseline:
            atom_counts = structure.get("atom_counts") or {"Ga": 0, "O": 0}
            energy_offset = (
                atom_counts.get("Ga", 0) * energy_diff["Ga"]
                + atom_counts.get("O", 0) * energy_diff["O"]
            )
            predicted_energy = predicted_energy - energy_offset

        lammps_energy_per_atom = predicted_energy / structure['num_atoms']
        config_type = structure['config_type']
        
        dft_energies.append(dft_energy_per_atom)
        lammps_energies.append(lammps_energy_per_atom)
        config_types.append(config_type)
    
    return np.array(dft_energies), np.array(lammps_energies), np.array(config_types)


def calculate_r2(y_true, y_pred):
    """Calculate R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def plot_energy_scatter(ax, dft_energies, lammps_energies, config_types, model_name, subplot_label=None,fontsize=10):
    """Plot energy scatter with config_type coloring"""
    default_colors = plt.cm.tab10.colors
    
    # Calculate statistics
    rmse = np.sqrt(np.mean((lammps_energies - dft_energies)**2))
    mae = np.mean(np.abs(lammps_energies - dft_energies))
    r2 = calculate_r2(dft_energies, lammps_energies)
    
    # Plot by config_type
    unique_configs = np.unique(config_types[config_types != None])
    for i, config_type in enumerate(unique_configs):
        if config_type is None:
            continue
        mask = config_types == config_type
        color = default_colors[i % len(default_colors)]
        ax.scatter(dft_energies[mask], lammps_energies[mask], 
                  alpha=0.6, s=20, color=color, edgecolor='none',
                  label=fr'$\mathit{{{config_type}}}$')
    
    # Handle None values
    none_mask = config_types == None
    if np.sum(none_mask) > 0:
        ax.scatter(dft_energies[none_mask], lammps_energies[none_mask], 
                  alpha=0.6, s=20, color='#cccccc', edgecolor='none',
                  label=f'unknown ({np.sum(none_mask)})')
    
    # Perfect prediction line
    min_val = min(dft_energies.min(), lammps_energies.min())
    max_val = max(dft_energies.max(), lammps_energies.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
           color='red', linewidth=1.5, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('DFT Energy (eV/atom)', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('LAMMPS Energy (eV/atom)', fontsize=fontsize, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if model_name == r"$\bf{tabGAP}$":
        ax.legend(loc='lower right', fontsize=fontsize-1)
    
    # Add subplot label (a), (b), (c), (d)
    if subplot_label:
        ax.text(-0.1, 1.1, subplot_label, transform=ax.transAxes,
               fontsize=fontsize+3, fontweight='bold', ha='left', va='top')
    
    # Add model name as text inside plot
    # ax.text(0.95, 0.05, model_name, transform=ax.transAxes,
    #        fontsize=fontsize, fontweight='bold', ha='right', va='bottom',
    #        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
    
    # Add statistics
    stats_text = model_name+f'\nRMSE={rmse*1000:.1f} meV/atom \nMAE={mae*1000:.1f} meV/atom'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=fontsize,
           bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})


def read_descriptor_file(filename):
    """Read descriptor data"""
    descriptors = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if '|' in parts[0]:
                parts = parts[0].split('|')[1:] + parts[1:]
            values = [float(x) for x in parts]
            descriptors.append(values)
    return np.array(descriptors)


def read_xyz_config_types(filename):
    """Read configuration types from xyz file"""
    config_types = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    idx = 0
    while idx < len(lines):
        try:
            n_atoms = int(lines[idx].strip())
            if idx + 1 < len(lines):
                properties_line = lines[idx + 1].strip()
                match = re.search(r'Config_type=(\S+)', properties_line)
                config_type = match.group(1) if match else "unknown"
                config_types.append(config_type)
            idx += n_atoms + 2
        except (ValueError, IndexError):
            break
    
    return config_types


def merge_config_types(config_types):
    """Merge: *GPa and v* -> augmented"""
    return ['augmented' if (ct.endswith('GPa') or ct.startswith('v')) else ct 
            for ct in config_types]


def perform_pca(descriptors):
    """Perform PCA analysis"""
    scaler = StandardScaler()
    descriptors_scaled = scaler.fit_transform(descriptors)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(descriptors_scaled)
    return transformed, pca


def plot_pca_panel(ax, pca_data, config_types, pca_obj, subplot_label=None,fontsize=10):
    """Plot PCA scatter"""
    
    # Configuration type mapping for cleaner legends
    mapping_table = {
        'bulk_beta_phase': r'$\beta$-phase',
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
        'melted_phase': r'melted',
        'isolated_atom': r'isolated Ga/O atoms',
        'close_3b_phase': r'close-3b phase',
        'amorphous_phase': r'amorphous',
    }

    unique_types = sorted(set(config_types))
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_types), 10)))
    if len(unique_types) > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_types), 20)))
    
    color_map = {ct: colors[i % len(colors)] for i, ct in enumerate(unique_types)}
    
    for config_type in unique_types:
        indices = [i for i, ct in enumerate(config_types) if ct == config_type]
        x = pca_data[indices, 0]
        y = pca_data[indices, 1]
        display_name = mapping_table.get(config_type, config_type)
        ax.scatter(x, y, c=[color_map[config_type]], 
                  label=f'{display_name}',
                  alpha=0.7, s=30, edgecolors='white', linewidths=0.5)
    
    var1 = pca_obj.explained_variance_ratio_[0] * 100
    var2 = pca_obj.explained_variance_ratio_[1] * 100
    
    ax.set_xlabel(f'Principal Component 1', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(f'Principal Component 2', fontsize=fontsize, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=fontsize-1, loc='best')
    
    # Add subplot label (d)
    if subplot_label:
        ax.text(-0.1, 1.1, subplot_label, transform=ax.transAxes,
               fontsize=fontsize+3, fontweight='bold', ha='left', va='top')


def run_lammps_for_model(test_xyz, forcefield, lammps_exe, workspace_root):
    """Run LAMMPS workflow for a model"""
    from run_lammps_workflow import (
        read_xyz_frames, save_frames_to_folders,
        convert_structures_to_lammps, create_symlinks_forcefield,
        create_symlinks_run_script, run_lammps_wrapper, generate_test_name
    )
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    
    ff_path = Path(forcefield)
    potential_type = "nep" if (ff_path.is_file() and ff_path.suffix == '.txt') else "tabgap"
    
    test_name = generate_test_name(test_xyz, forcefield)
    raw_data_dir = workspace_root / "run" / "raw_data" / test_name
    
    # Check if already exists
    if raw_data_dir.exists():
        subdirs = [d for d in raw_data_dir.iterdir() if d.is_dir()]
        log_exists = sum(1 for d in subdirs if (d / "lammps.log").exists())
        if log_exists > 0:
            print(f"  Using existing results: {test_name} ({log_exists} logs)")
            return raw_data_dir
    
    print(f"  Running LAMMPS: {test_name}...")
    
    frames = read_xyz_frames(str(test_xyz))
    save_frames_to_folders(frames, str(raw_data_dir))
    convert_structures_to_lammps(str(raw_data_dir))
    
    run_script = "scripts/run_nep.in" if potential_type == "nep" else "scripts/run_gap.in"
    create_symlinks_forcefield(forcefield, str(raw_data_dir), potential_type)
    create_symlinks_run_script(run_script, str(raw_data_dir))
    
    lammps_path = Path(lammps_exe).resolve()
    subdirs = [d for d in raw_data_dir.iterdir() if d.is_dir() and (d / "run.in").exists()]
    subdirs = sorted(subdirs)
    
    n_cores = os.cpu_count()
    tasks = [(lammps_path, subdir, i+1, len(subdirs)) for i, subdir in enumerate(subdirs)]
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_task = {executor.submit(run_lammps_wrapper, task): task for task in tasks}
        for future in as_completed(future_to_task):
            try:
                _, _, success, _ = future.result()
                if success:
                    success_count += 1
            except Exception:
                pass
    
    print(f"    Completed: {success_count}/{len(subdirs)}")
    return raw_data_dir


def main():
    workspace_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("Comprehensive Analysis: 3 Models + PCA")
    print("=" * 80)
    print(f"NEP test xyz: {test_xyz}")
    if tabgap_test_xyz:
        print(f"tabGAP test xyz: {tabgap_test_xyz}")
    print("=" * 80)

    if len(models) != 3 or len(names) != 3:
        print("Error: This script expects exactly 3 models and 3 names.")
        return 1

    if len(models) != len(names):
        print("Error: models and names must have the same length.")
        return 1
    
    # Collect predictions
    print("\n[1/3] Collecting predictions...")
    model_predictions = []
    
    if not skip_run:
        for i, (model, name) in enumerate(zip(models, names)):
            print(f"\n  Model {i+1}/3: {name}")
            
            # Detect potential type
            ff_path = Path(model)
            is_tabgap = not (ff_path.is_file() and ff_path.suffix == '.txt')
            potential_type = "tabgap" if is_tabgap else "nep"
            
            # Choose appropriate test xyz
            if is_tabgap and tabgap_test_xyz:
                model_test_xyz = tabgap_test_xyz
                print(f"    Using tabGAP test xyz: {model_test_xyz}")
            else:
                model_test_xyz = test_xyz
                print(f"    Using NEP test xyz: {model_test_xyz}")
            
            # Read structures for this model
            structures = read_test_xyz_structures(model_test_xyz)
            print(f"    Loaded {len(structures)} structures")
            
            # Run LAMMPS and collect predictions
            data_dir = run_lammps_for_model(model_test_xyz, model, lammps_exe, workspace_root)
            dft_e, lmp_e, configs = collect_predictions(data_dir, structures, potential_type)
            model_predictions.append((dft_e, lmp_e, configs))
    else:
        from run_lammps_workflow import generate_test_name
        for i, (model, name) in enumerate(zip(models, names)):
            print(f"\n  Model {i+1}/3: {name}")
            
            # Detect potential type
            ff_path = Path(model)
            is_tabgap = not (ff_path.is_file() and ff_path.suffix == '.txt')
            potential_type = "tabgap" if is_tabgap else "nep"
            
            # Choose appropriate test xyz
            if is_tabgap and tabgap_test_xyz:
                model_test_xyz = tabgap_test_xyz
                print(f"    Using tabGAP test xyz: {model_test_xyz}")
            else:
                model_test_xyz = test_xyz
                print(f"    Using NEP test xyz: {model_test_xyz}")
            
            # Read structures for this model
            structures = read_test_xyz_structures(model_test_xyz)
            print(f"    Loaded {len(structures)} structures")
            
            # Collect predictions from existing results
            test_name = generate_test_name(model_test_xyz, model)
            data_dir = workspace_root / "run" / "raw_data" / test_name
            print(f"    Using existing data: {test_name}")
            dft_e, lmp_e, configs = collect_predictions(data_dir, structures, potential_type)
            model_predictions.append((dft_e, lmp_e, configs))
    
    # PCA analysis
    print("\n[2/3] Performing PCA...")
    descriptors = read_descriptor_file(descriptor_file)
    print(f"  Loaded {len(descriptors)} descriptor vectors")
    
    config_types = read_xyz_config_types(descriptor_xyz)
    print(f"  Loaded {len(config_types)} config types")
    
    if merge_types:
        config_types = merge_config_types(config_types)
        print(f"  Merged config types (*GPa, v* → augmented)")
    
    min_len = min(len(descriptors), len(config_types))
    pca_data, pca_obj = perform_pca(descriptors[:min_len])
    config_types = config_types[:min_len]
    
    var1 = pca_obj.explained_variance_ratio_[0] * 100
    var2 = pca_obj.explained_variance_ratio_[1] * 100
    print(f"  PCA variance: PC1={var1:.1f}%, PC2={var2:.1f}%, Total={var1+var2:.1f}%")
    
    # Create figure
    print("\n[3/3] Generating figure...")
    
    from matplotlib.gridspec import GridSpec
    
    # Set font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # Ultra-fine grid settings
    n = 100
    x0 = 10 * n
    y0 = 8 * n
    dx = int(n * 2.5)  # Horizontal spacing
    dy = int(n * 2.0)  # Vertical spacing
    
    M = 2 * x0 + dx
    N = 2 * y0 + dy
    
    figsize = 10
    fontsize = 10
    fig = plt.figure(figsize=(figsize, N/(M/figsize)))
    gs = GridSpec(N, M, figure=fig, width_ratios=np.ones(M), height_ratios=np.ones(N))
    
    # Create subplots manually
    axes = []
    
    # (a) Top-Left
    axes.append(fig.add_subplot(gs[0:y0, 0:x0]))
    
    # (b) Top-Right
    axes.append(fig.add_subplot(gs[0:y0, x0+dx:2*x0+dx]))
    
    # (c) Bottom-Left
    axes.append(fig.add_subplot(gs[y0+dy:2*y0+dy, 0:x0]))
    
    # (d) Bottom-Right
    axes.append(fig.add_subplot(gs[y0+dy:2*y0+dy, x0+dx:2*x0+dx]))
    
    # Subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    
    # Plot 3 models
    for i in range(3):
        dft_e, lmp_e, configs = model_predictions[i]
        print(f"  Panel {i+1}: {names[i]} ({len(dft_e)} points)")
        plot_energy_scatter(axes[i], dft_e, lmp_e, configs, names[i], subplot_labels[i],fontsize=fontsize)
    
    # Plot PCA
    print(f"  Panel 4: PCA Analysis ({len(pca_data)} points)")
    plot_pca_panel(axes[3], pca_data, config_types, pca_obj, subplot_labels[3],fontsize=fontsize)
    
    # plt.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"\n✓ Figure saved: {output_file}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
