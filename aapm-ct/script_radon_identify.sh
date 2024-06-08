#!/bin/bash

#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -P 90000001
#PBS -N forward 

# Load necessary modules
module load cudatoolkit/10.1
module load python/3.8.3
module load pytorch/1.6.0
module load torchvision/0.7.0
module load matplotlib/3.1.3
module load numpy/1.18.1
module load pandas/1.0.5
module load tqdm/4.46.0

# Change to the directory from which the job was submitted
cd ${PBS_O_WORKDIR}

# Run the Python script
python config.py
python script_radon_identify.py