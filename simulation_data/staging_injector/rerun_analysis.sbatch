#!/bin/bash -l

#SBATCH -t 00:30:00
#SBATCH -N 10
#SBATCH -A m558
#SBATCH -q regular
#SBATCH -C cpu
# 8 cores per chiplet, 2x SMP
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive

# Source environment
source $HOME/perlmutter_gpu_warpx.profile

# Run optimas scan
srun --cpu-bind=cores python rerun_analysis.py --evaluations_directory=${1}
