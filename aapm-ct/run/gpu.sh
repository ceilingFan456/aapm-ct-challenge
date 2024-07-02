#!/bin/bash

#PBS -q normal
#PBS -j oe
#PBS -l select=1:ngpus=1
#PBS -l walltime=01:00:00
#PBS -N forward 


echo "Job started at: $(date)"
nvidia-smi
