#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=360:00:00
#SBATCH --mem-per-cpu=10240
#SBATCH --mail-type=END

source ~/miniconda3/etc/profile.d/conda.sh
conda activate forl_env

python -u main.py 64 32 16
