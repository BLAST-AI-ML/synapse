#!/bin/bash --login

# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate ml-training

# Execute the provided command
exec "$@"
