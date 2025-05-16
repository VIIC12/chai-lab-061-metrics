#!/usr/bin/env python3
"""
Author:     Tom U. Schlegel
Date:       2025-05-16
Name:       calc_chaiai_metrics
Info:       Prints ChaiAI metrics as json, plot PAE and PDE map.
"""

#!/usr/bin/env python3
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import argparse

def detect_mode(npz_file: str) -> str:
    """Detect whether the NPZ file is from scfv or separate chain mode.
    
    Args:
        npz_file: Path to NPZ file containing ChaiAI metrics
        
    Returns:
        str: Either 'scfv' or 'separate' based on chain count
    """
    data = np.load(npz_file)
    if 'per_chain_plddt' in data:
        chain_count = len(data['per_chain_plddt'][0])
        return 'separate' if chain_count == 3 else 'scfv'
    return 'scfv'  # Default to scfv if can't detect

def plot_pae(pae_values: np.ndarray, output_path: str, name: str = "") -> None:
    """Creates and saves a PAE heatmap plot.
    
    Args:
        pae_values: 2D numpy array of PAE values
        output_path: Path where to save the plot
        name: Optional name to include in the plot title
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pde(pde_values: np.ndarray, output_path: str, name: str = "") -> None:
    """Creates and saves a PDE heatmap plot.
    
    Args:
        pde_values: 2D numpy array of PDE values
        output_path: Path where to save the plot
        name: Optional name to include in the plot title
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_metrics(npz_file: str, save: bool = False, output_dir: Path = None, plot: bool = False) -> None:
    """Extract metrics from ChaiAI NPZ file and print as JSON.
    
    Args:
        npz_file: Path to NPZ file containing ChaiAI metrics
        save: Whether to save results to a file
        output_dir: Optional custom output directory for plots and JSON
        plot: Whether to save PAE/PDE plots
    """
    data = np.load(npz_file)
    mode = detect_mode(npz_file)
    #print(f"Detected mode: {mode}")

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
    ranking_score_complex = float(data['ranking_score_complex'][0]) if 'ranking_score_complex' in data else None
    fraction_disordered = float(data['fraction_disordered_complex']) if 'fraction_disordered_complex' in data else None
    fraction_disordered_binder = float(data['fraction_disordered_binder']) if 'fraction_disordered_binder' in data else None
    
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

    # Ranking scores
    ranking_score_binder = float(data['ranking_score_binder'][0]) if 'ranking_score_binder' in data else None

    # PDE metrics
    pde_binder = float(data['pde_binder']) if 'pde_binder' in data else None
    pde_targets = float(data['pde_targets']) if 'pde_targets' in data else None

    # Save PAE and PDE plots if plot flag is set
    if plot:
        if 'pae' in data:
            pae_output = output_dir / f"{base_name}_pae.png"
            plot_pae(data['pae'], str(pae_output), base_name)
        
        if 'pde' in data:
            pde_output = output_dir / f"{base_name}_pde.png"
            plot_pde(data['pde'], str(pde_output), base_name)

    # Define the output structure
    results = {
        "complex": {
            'aggregate_score': round(aggregate_score, 4) if aggregate_score is not None else None,
            'plddt_complex': round(complex_plddt, 4) if complex_plddt is not None else None,
            'pTM_complex': round(complex_ptm, 4) if complex_ptm is not None else None,
            'iptm_ptm': round(iptm_ptm, 4) if iptm_ptm is not None else None,
            'ranking_score_complex': round(ranking_score_complex, 4) if ranking_score_complex is not None else None,
            'frac_disordered_complex': round(fraction_disordered, 4) if fraction_disordered is not None else None,
            'atom_has_clash': bool(has_clash),
            'pae_complex': round(pae_complex, 4) if pae_complex is not None else None
        },
        "binder": {
            'ranking_score_binder': round(ranking_score_binder, 4) if ranking_score_binder is not None else None,
            'plddt_binder': round(float(chain_plddt[0]), 4) if chain_plddt is not None else None,
            'pTM_binder': round(float(chain_ptm[0]), 4) if chain_ptm is not None else None,
            'pae_binder': round(pae_binder, 4) if pae_binder is not None else None,
            'pae_min_binder': round(binder_pae_min, 4) if binder_pae_min is not None else None,
            'pde_binder': round(pde_binder, 4) if pde_binder is not None else None,
            'frac_disordered_binder': round(fraction_disordered_binder, 4) if fraction_disordered_binder is not None else None
        },
        "targets": {
            'plddt_targets': round(float(np.mean(chain_plddt[1:])), 4) if chain_plddt is not None and len(chain_plddt) > 1 else None,
            'pTM_targets': round(float(np.mean(chain_ptm[1:])), 4) if chain_ptm is not None and len(chain_ptm) > 1 else None,
            'pae_targets': round(pae_targets, 4) if pae_targets is not None else None,
            'pae_min_targets': round(target_pae_min, 4) if target_pae_min is not None else None,
            'pde_targets': round(pde_targets, 4) if pde_targets is not None else None
        },
        "interaction": {
            'interface_ptm': round(interface_ptm, 4) if interface_ptm is not None else None,
            'pae_interaction': round(pae_interaction, 4) if pae_interaction is not None else None,
            'pae_interactions_per_target': [round(float(x), 4) for x in pae_interactions_per_target] if pae_interactions_per_target is not None else None
        }
    }

    # Remove None values from the results
    results = {k: {ik: iv for ik, iv in v.items() if iv is not None} for k, v in results.items()}
    results = {k: v for k, v in results.items() if v}

    # Print all scores except PDE and PAE arrays
    if save:
        output_json = output_dir / f"{base_name}_metrics.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)
    else:
        print(json.dumps(results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file", help="Path to NPZ file containing ChaiAI metrics")
    parser.add_argument("--save", action="store_true", help="Save results to a JSON file")
    parser.add_argument("--output", type=str, help="Custom output directory for plots and JSON")
    parser.add_argument("--plot", action="store_true", help="Save PAE and PDE plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    get_metrics(args.npz_file, args.save, output_dir, args.plot)