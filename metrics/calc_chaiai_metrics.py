#!/usr/bin/env python3
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import argparse

# PyRosetta imports for clash calculation
from pyrosetta import init, pose_from_file
from pyrosetta.rosetta.core.pose import conf2pdb_chain
from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import PerResidueClashMetric
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector

plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 5,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

# --- Clash calculation helpers (copied from calc_boltz_metrics.py) ---
def _calc_intra_clash_for_chain(pose, chain_id: str):
    """Return total number of intrachain clashes for a chain."""
    clash_metric = PerResidueClashMetric()
    clash_metric.set_residue_selector(ChainSelector(chain_id))
    per_residue_clash = clash_metric.calculate(pose)
    return np.sum([per_residue_clash[key] for key in per_residue_clash])

def _calc_inter_clash_score(pose, binder_chain: str, target_chain: str):
    """Return clash_score between two chains."""
    clash_metric = PerResidueClashMetric()
    clash_metric.set_residue_selector(ChainSelector(binder_chain))
    clash_metric.set_secondary_residue_selector(ChainSelector(target_chain))
    inter_clash = clash_metric.calculate(pose)
    return np.sum([inter_clash[key] for key in inter_clash])
# --- End clash calculation helpers ---

def plot_chain_interactions(chain_data: np.ndarray, output_path: str, name: str = "", uid: str = "") -> None:
    """Creates and saves a chain-chain interaction heatmap plot."""
    fontsize_header = 18
    fontsize_labels = 15

    # Convert chain data to matrix (remove the extra dimension from ChaiAI output)
    matrix = chain_data[0]  # Shape is (1, n, n), we want (n, n)

    plt.figure(figsize=(7, 5))
    
    plt.imshow(matrix, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='iPTM score')
    plt.suptitle('Chain-Chain Interactions', fontsize=fontsize_header)
    if name:
        plt.title(name, fontsize=fontsize_header)
    
    # Set ticks and labels with binder/target designation
    chain_labels = []
    for i in range(matrix.shape[0]):
        if i == 0:
            chain_labels.append('Binder')
        else:
            chain_labels.append(f'Target {i}')
    
    plt.xticks(range(len(chain_labels)), chain_labels, fontsize=fontsize_labels)
    plt.yticks(range(len(chain_labels)), chain_labels, fontsize=fontsize_labels)
    
    # Add text annotations with adjusted color threshold
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f'{matrix[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='white' if matrix[i, j] < 0.7 else 'black',  # Adjusted threshold for better readability
                    fontsize=12)
    
    # Make the border thicker
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.3)
    
    if uid:
        path = Path(output_path)
        output_path = str(path.parent / f"{path.stem}_{uid}{path.suffix}")
        
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()

def plot_pae(pae_values: np.ndarray, output_path: str, name: str = "", uid: str = "") -> None:
    """Creates and saves a PAE heatmap plot.
    
    Args:
        pae_values: 2D numpy array of PAE values
        output_path: Path where to save the plot
        name: Optional name to include in the plot title
        uid: Optional unique identifier to add to plot filename
    """
    fontsize_header = 18
    fontsize_labels = 15

    plt.figure(figsize=(7, 5))
    plt.imshow(pae_values, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Expected position error (Å)')
    plt.suptitle('Predicted Aligned Error (pAE)', fontsize=fontsize_header)
    if name:
        plt.title(name, fontsize=fontsize_header)
    plt.xlabel('Scored residue', fontsize=fontsize_labels)
    plt.ylabel('Aligned residue', fontsize=fontsize_labels)
    
    # Make the border thicker
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.3)
    
    if uid:
        path = Path(output_path)
        output_path = str(path.parent / f"{path.stem}_{uid}{path.suffix}")
        
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()

def plot_pde(pde_values: np.ndarray, output_path: str, name: str = "", uid: str = "") -> None:
    """Creates and saves a PDE heatmap plot.
    
    Args:
        pde_values: 2D numpy array of PDE values
        output_path: Path where to save the plot
        name: Optional name to include in the plot title
        uid: Optional unique identifier to add to plot filename
    """
    fontsize_header = 18
    fontsize_labels = 15

    plt.figure(figsize=(7, 5))
    plt.imshow(pde_values, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Expected distance error (Å)')
    plt.suptitle('Predicted Distance Error (pDE)', fontsize=fontsize_header)
    if name:
        plt.title(name, fontsize=fontsize_header)
    plt.xlabel('Residue i', fontsize=fontsize_labels)
    plt.ylabel('Residue j', fontsize=fontsize_labels)
    
    # Make the border thicker
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.3)
    
    if uid:
        path = Path(output_path)
        output_path = str(path.parent / f"{path.stem}_{uid}{path.suffix}")
        
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()

def get_metrics(npz_file: str, pdb_file: str, save: bool = False, output_dir: Path = None, plot: bool = False, uid: str = "") -> None:
    """Extract metrics from ChaiAI NPZ file and print as JSON.
    
    Args:
        npz_file: Path to NPZ file containing ChaiAI metrics
        pdb_file: Path to PDB file for clash calculation
        save: Whether to save results to a file
        output_dir: Optional custom output directory for plots and JSON
        plot: Whether to save PAE/PDE plots
        uid: Optional unique identifier to add to output filenames
    """
    data = np.load(npz_file)

    # Get the output directory for plots
    if output_dir is None:
        output_dir = Path(npz_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    base_name = Path(npz_file).stem

    # Extract all metrics from data
    aggregate_score = float(data['aggregate_score'][0]) if 'aggregate_score' in data else None
    complex_plddt = float(data['complex_plddt'][0]) if 'complex_plddt' in data else None
    complex_ptm = float(data['complex_ptm'][0]) if 'complex_ptm' in data else None
    interface_ptm = float(data['interface_ptm'][0]) if 'interface_ptm' in data else None
    iptm_ptm = float(data['iptm_ptm'][0]) if 'iptm_ptm' in data else None

    fraction_disordered = float(data['fraction_disordered_complex']) if 'fraction_disordered_complex' in data else None
    fraction_disordered_binder = float(data['fraction_disordered_binder']) if 'fraction_disordered_binder' in data else None

    #! General clash detection, not for ranking_score calculation
    # Handle clash detection
    has_inter_chain_clashes = bool(data['has_inter_chain_clashes'][0]) if 'has_inter_chain_clashes' in data else None
    chain_chain_clashes = data['chain_chain_clashes'][0] if 'chain_chain_clashes' in data else None
    
    # Determine if there are any clashes based on both conditions
    has_clash = False
    if has_inter_chain_clashes is not None and chain_chain_clashes is not None:
        has_clash = has_inter_chain_clashes or np.any(chain_chain_clashes > 0)


    # Per chain metrics
    chain_plddt = data['per_chain_plddt'][0] if 'per_chain_plddt' in data else None
    chain_ptm = data['per_chain_ptm'][0] if 'per_chain_ptm' in data else None

    # PAE metrics
    pae_complex = float(data['pae_complex']) if 'pae_complex' in data else None
    pae_binder = float(data['pae_binder']) if 'pae_binder' in data else None
    pae_targets = float(data['pae_targets']) if 'pae_targets' in data else None
    pae_interaction = float(data['pae_interaction']) if 'pae_interaction' in data else None
    pae_interactions_per_target = data['pae_interactions_per_target'] if 'pae_interactions_per_target' in data else None
    binder_pae_min = float(data['binder_pae_min']) if 'binder_pae_min' in data else None
    target_pae_min = float(data['target_pae_global_min']) if 'target_pae_global_min' in data else None

    # PDE metrics
    pde_binder = float(data['pde_binder']) if 'pde_binder' in data else None
    pde_targets = float(data['pde_targets']) if 'pde_targets' in data else None

    # --- CLASH CALCULATION USING PDB ---
    # Initialize PyRosetta
    init('-mute all', silent=True)
    pose = pose_from_file(str(pdb_file))
    pose_chains = pose.split_by_chain()
    n_chains = len(pose_chains)
    if n_chains < 2:
        raise ValueError('At least 2 chains (binder / target) are required.')
    binder_pose = pose_chains[1]
    target_poses = [pose_chains[i] for i in range(2, n_chains + 1)]
    chains = {k: v for k, v in conf2pdb_chain(pose).items()}
    binder_chain = chains[1]
    target_chains = [chains[i] for i in range(2, len(chains) + 1)]
    intra_clash_binder = _calc_intra_clash_for_chain(pose, binder_chain)
    
    #! Added
    intra_clash_targets = sum(_calc_intra_clash_for_chain(pose, chain) for chain in target_chains)
    inter_clash = sum(_calc_inter_clash_score(pose, binder_chain, target_chain) for target_chain in target_chains)
    total_clash = intra_clash_binder + intra_clash_targets + inter_clash

    has_binder_clash = (
        intra_clash_binder > 100 or 
        intra_clash_binder > (binder_pose.total_residue() / 2)
    )
    has_target_clash = any(
        (_calc_intra_clash_for_chain(pose, chain) > 100 or
         _calc_intra_clash_for_chain(pose, chain) > (pose.total_residue() / 2))
        for chain in target_chains
    )
    has_complex_clash = has_binder_clash or has_target_clash

    # Ranking scores, calculate extra
    ranking_score_binder = 0.8 * interface_ptm + 0.2 * chain_ptm[0] + 0.5 * fraction_disordered_binder - 100 * has_binder_clash
    ranking_score_complex = 0.8 * interface_ptm + 0.2 * complex_ptm + 0.5 * fraction_disordered - 100 * has_complex_clash


    # --- END CLASH CALCULATION ---

    # Save PAE, PDE and chain interactions plots if plot flag is set
    if plot:
        if 'pae' in data:
            pae_output = output_dir / f"{base_name}_pae.png"
            plot_pae(data['pae'], str(pae_output), base_name, uid)
        
        if 'pde' in data:
            pde_output = output_dir / f"{base_name}_pde.png"
            plot_pde(data['pde'], str(pde_output), base_name, uid)
            
        if 'per_chain_pair_iptm' in data:
            chain_interactions_output = output_dir / f"{base_name}_chain_interactions.png"
            plot_chain_interactions(data['per_chain_pair_iptm'], str(chain_interactions_output), base_name, uid)

    # Define the output structure
    results = {
        "complex": {
            'aggregate_score': round(aggregate_score, 4) if aggregate_score is not None else None,
            'plddt_complex': round(complex_plddt, 4) if complex_plddt is not None else None,
            'pTM_complex': round(complex_ptm, 4) if complex_ptm is not None else None,
            'iptm_ptm': round(iptm_ptm, 4) if iptm_ptm is not None else None,
            'ranking_score_complex': round(ranking_score_complex, 4) if ranking_score_complex is not None else None,
            'frac_disordered_complex': round(fraction_disordered, 4) if fraction_disordered is not None else None,
            'any_clash': bool(has_clash),
            'total_clash_number': int(total_clash),
            'pae_complex': round(pae_complex, 4) if pae_complex is not None else None
        },
        "binder": {
            'ranking_score_binder': round(ranking_score_binder, 4) if ranking_score_binder is not None else None,
            'plddt_binder': round(float(chain_plddt[0]), 4) if chain_plddt is not None else None,
            'pTM_binder': round(float(chain_ptm[0]), 4) if chain_ptm is not None else None,
            'pae_binder': round(pae_binder, 4) if pae_binder is not None else None,
            'pae_min_binder': round(binder_pae_min, 4) if binder_pae_min is not None else None,
            'pde_binder': round(pde_binder, 4) if pde_binder is not None else None,
            'frac_disordered_binder': round(fraction_disordered_binder, 4) if fraction_disordered_binder is not None else None,
            'intra_clash_number': int(intra_clash_binder),
            'any_clash_treshold': bool(has_binder_clash)
        },
        "targets": {
            'plddt_targets': round(float(np.mean(chain_plddt[1:])), 4) if chain_plddt is not None and len(chain_plddt) > 1 else None,
            'pTM_targets': round(float(np.mean(chain_ptm[1:])), 4) if chain_ptm is not None and len(chain_ptm) > 1 else None,
            'pae_targets': round(pae_targets, 4) if pae_targets is not None else None,
            'pae_min_targets': round(target_pae_min, 4) if target_pae_min is not None else None,
            'pde_targets': round(pde_targets, 4) if pde_targets is not None else None,
            'intra_clash_number': int(intra_clash_targets),
            'any_clash_treshold': bool(has_target_clash)
        },
        "interaction": {
            'interface_ptm': round(interface_ptm, 4) if interface_ptm is not None else None,
            'pae_interaction': round(pae_interaction, 4) if pae_interaction is not None else None,
            'pae_interactions_per_target': "; ".join(str(round(float(x), 4)) for x in pae_interactions_per_target) if pae_interactions_per_target is not None else None,
            'inter_clash_number': int(inter_clash)
        }
    }

    # Remove None values from the results
    results = {k: {ik: iv for ik, iv in v.items() if iv is not None} for k, v in results.items()}
    results = {k: v for k, v in results.items() if v}

    # Print all scores except PDE and PAE arrays
    if save:
        # Add uid to output filename if provided
        if uid:
            output_json = output_dir / f"{base_name}_metrics_{uid}.json"
        else:
            output_json = output_dir / f"{base_name}_metrics.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file", help="Path to NPZ file containing ChaiAI metrics")
    parser.add_argument("pdb_file", help="Path to PDB file for clash calculation")
    parser.add_argument("--save", action="store_true", help="Save results to a JSON file")
    parser.add_argument("--output", type=str, help="Custom output directory for plots and JSON")
    parser.add_argument("--plot", action="store_true", help="Save PAE and PDE plots")
    parser.add_argument("--uid", type=str, default="", help="Unique identifier to add to output filenames")
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    get_metrics(args.npz_file, args.pdb_file, args.save, output_dir, args.plot, args.uid)