#!/bin/bash -l

# Copyright 2021-2023 Axel Huebl, Kevin Gott
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:30:00
#SBATCH -N 4
#SBATCH -J WarpX
#    note: <proj> must end on _g
#SBATCH -A m4546_g
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
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j

# executable & inputs file or python interpreter & PICMI script here
EXE=./warpx.2d
INPUTS=inputs_2d

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU

# threads for OpenMP and threaded compressors per MPI rank
#   note: 16 avoids hyperthreading (32 virtual cores, 16 physical)
export SRUN_CPUS_PER_TASK=16

# GPU-aware MPI optimizations
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=1"

#export MAX_STEP=150000

# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
srun --cpu-bind=cores bash -c "
    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
    ${EXE} ${INPUTS} ${GPU_AWARE_MPI}" \
  > output_${SLURM_JOBID}.txt
