#!/bin/bash --login

# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate gui

# Execute the provided command
exec "$@"
