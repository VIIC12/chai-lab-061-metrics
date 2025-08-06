import sys
from pathlib import Path
import logging
import numpy as np
import torch
import shutil
import argparse

logging.basicConfig(level=logging.INFO)  # control verbosity

from chai_lab.chai1 import run_inference

def parse_args():
    parser = argparse.ArgumentParser(description='Run ChaiAI inference')
    parser.add_argument('fasta_file', type=str, help='Path to input FASTA file')
    parser.add_argument('output_dir', type=str, help='Path to output directory')
    parser.add_argument('--use_msa', action='store_true', help='Use MSA server for inference')
    parser.add_argument('--msa', type=str, help='Path to MSA directory')
    parser.add_argument('--constraints', type=str, help='Path to constraints file')
    parser.add_argument('--fold_intensify', action='store_true', help='Fold intensify')
    return parser.parse_args()

def main():
    args = parse_args()

    # Inference expects an empty directory; enforce this
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        logging.warning(f"Removing old output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Prepare inference parameters
    inference_params = {
        'fasta_file': Path(args.fasta_file),
        'output_dir': output_dir,
        'device': "cuda:0",
    }

    # MSA server
    if args.use_msa:
        inference_params.update({
            'use_templates_server': True,
            'use_msa_server': True,
        })
    
    # Custom MSA
    if args.msa:
        inference_params['msa_directory'] = Path(args.msa)
    
    # Constraints
    if args.constraints:
        inference_params['constraint_path'] = args.constraints

    # Fold intensify
    if args.fold_intensify:
        inference_params.update({
            'num_trunk_recycles': 6,
            'num_diffn_timesteps': 400,
        })

    # Run inference
    output = run_inference(**inference_params)

if __name__ == "__main__":
    main()