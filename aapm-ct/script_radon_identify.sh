#!/bin/bash

#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N forward 

# Load available modules
module load python/3.8.13
module load cuda/11.6.2
module load pytorch/1.11.0-py3-gpu

cd ${PBS_O_WORKDIR}

# Create a virtual environment
python -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install matplotlib numpy pandas torchvision tqdm

# Run the Python script
python config.py
python script_radon_identify.py

# Deactivate the virtual environment
deactivate
