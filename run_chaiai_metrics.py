import sys
from pathlib import Path
import logging
import numpy as np
import torch
import shutil

logging.basicConfig(level=logging.INFO)  # control verbosity

from chai_lab.chai1 import run_inference

# Inference expects an empty directory; enforce this
output_dir = Path(sys.argv[2])
if output_dir.exists():
    logging.warning(f"Removing old output directory: {output_dir}")
    shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True)


# Expect binder chain to come first
output_paths = run_inference(
    fasta_file=Path(sys.argv[1]),
    output_dir=output_dir,
    device=torch.device("cuda:0")
)