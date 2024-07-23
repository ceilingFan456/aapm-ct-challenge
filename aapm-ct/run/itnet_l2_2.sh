#!/bin/bash

#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00
#PBS -N itnet_l2_2

# Log the date and time
echo "Job started at: $(date)"

module load miniforge3
conda activate myenv

cd /home/users/nus/e0271228/aapm-ct-challenge/aapm-ct/

python config.py
python script_train_itnet_post_memory_2.py
