#!/bin/bash
# Setup deforest environment on Aqua (run ONCE, on a GPU interactive node)
#
# Usage:
#   ssh maitreje@aqua.qut.edu.au
#   qsub -I -S /bin/bash -l select=1:ncpus=6:ngpus=1:mem=34gb -l walltime=02:00:00
#   bash ~/Deforestation/aqua/setup_env.sh

set -e

echo "=== Setting up deforest environment on Aqua ==="

module load Miniconda3/24.9.2-0 || module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"

# Create environment
conda create -n deforest python=3.11 -y
conda activate deforest

# Core packages
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install lightning

# TerraTorch (Prithvi-EO)
python -m pip install terratorch

# LoRA
python -m pip install peft

# Data
python -m pip install h5py pandas numpy scipy rasterio

# Drive download
python -m pip install gdown

# Verify
echo ""
echo "=== Verification ==="
nvidia-smi
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')

import terratorch
print(f'TerraTorch: OK')

import lightning as L
print(f'Lightning: {L.__version__}')

import h5py
print(f'h5py: {h5py.__version__}')

print('\\nAll good!')
"

echo ""
echo "=== Done ==="
echo "Next:"
echo "  1. (on local) rsync code to Aqua"
echo "  2. (on Aqua)  bash aqua/full_pipeline.sh   # download tiles + chip + ready to train"
echo "  3. (on Aqua)  qsub aqua/train_prithvi.pbs  # submit training"
