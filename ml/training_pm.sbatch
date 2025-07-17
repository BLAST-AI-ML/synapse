#!/bin/bash -l

# Copyright 2024-2025 Revathi Jambunathan, Axel Huebl
#
# This file is part of IFE Superfacility LDRD
#
# License: BSD-3-Clause-LBNL

#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH -J trainsf
#SBATCH -A m558
#SBATCH -q realtime
#SBATCH --constraint=gpu
# ideally single:1, but NERSC cgroups issue
#SBATCH --ntasks-per-node=1
#SBATCH -o /global/cfs/cdirs/m558/superfacility/model_training/logs/sf.o%j
#SBATCH -e /global/cfs/cdirs/m558/superfacility/model_training/logs/sf.e%j

experiment=${1}  # e.g., "ip2"
model=${2}  # e.g., "NN"

if [[ -z "${experiment}" ]]; then
    echo "Must pass the experiment name/tag as a command line argument"
fi

# pull container file ahead of time from a login node!
#docker login registry.nersc.gov
#podman-hpc pull registry.nersc.gov/m558/superfacility/ml-training:latest

# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
srun podman-hpc run --gpu -v $HOME/db.profile:/root/db.profile --rm -it ml-training:latest python -u /app/ml/train_model.py --experiment ${experiment} --model ${model}
