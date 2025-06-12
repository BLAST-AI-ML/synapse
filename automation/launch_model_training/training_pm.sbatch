#!/bin/bash -l

# Copyright 2024-2025 Revathi Jambunathan
#
# This file is part of IFE Superfacility LDRD
#
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -J testsf
#SBATCH -A m558
#SBATCH -q debug
#SBATCH --constraint=cpu
# ideally single:1, but NERSC cgroups issue
#SBATCH --ntasks-per-node=1
#SBATCH -o /global/cfs/cdirs/m558/superfacility/model_training/logs/sf.o%j
#SBATCH -e /global/cfs/cdirs/m558/superfacility/model_training/logs/sf.e%j

module load python
conda activate /global/cfs/cdirs/m558/superfacility/model_training/env

# threads for OpenMP and threaded compressors per MPI rank
#   note: 16 avoids hyperthreading (32 virtual cores, 16 physical)
export OMP_NUM_THREADS=1

# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
srun python /global/cfs/cdirs/m558/superfacility/model_training/src/mongo_NN.py
