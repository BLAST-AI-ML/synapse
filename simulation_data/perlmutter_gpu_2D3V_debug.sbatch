#!/bin/bash -l

### Use this file for debug runs
# Change the following settings to ensure simulations run on less resources:
# `run_grid_scan.py`:
#     - change `sim_workers` to 4
#     - set `n_gpus` to 4 (per worker)
#     - reduce the number of total simulations to ,e.g., just four
# `template_inputs_2d`:
#     - reduce the number of cells to something easily manageable (e.g., 256 384)
#     - change `num_procs` to 2 2
###

# Copyright 2021-2023 Axel Huebl, Kevin Gott
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH -J WarpX
#    note: <proj> must end on _g
#SBATCH -A m3239_g
#SBATCH -q debug
# A100 40GB (most nodes)
#SBATCH -C gpu
# A100 80GB (256 nodes)
#S BATCH -C gpu&hbm80g
#SBATCH --exclusive
# ideally single:1, but NERSC cgroups issue
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o WarpX_grid_scan.o%j
#SBATCH -e WarpX_grid_scan.e%j

# executable & inputs file or python interpreter & PICMI script here
EXE=python
INPUTS=run_grid_scan.py

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU

# threads for OpenMP and threaded compressors per MPI rank
#   note: 16 avoids hyperthreading (32 virtual cores, 16 physical)
export SRUN_CPUS_PER_TASK=16

# GPU-aware MPI optimizations
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
# Stephen Hudson has an open issue with NERSC
#   Starting with srun does not work immediately
#srun --cpu-bind=cores bash -c "
#    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
#    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}" \
#  > output_${SLURM_JOBID}.txt

${EXE} ${INPUTS}
