#!/bin/bash --login

# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate synapse-gui

# Give dashboard libraries that rely on Path.home() a stable home in Spin.
export HOME=/app/dashboard

# amsc-client's default Globus token cache is ~/.amsc/credentials.json.
# Ensure the directory exists at runtime; the Dockerfile created it as
# mode 1777 so the random UID Spin assigns can write into it.
mkdir -p "${HOME}/.amsc" 2>/dev/null || true

# Execute the provided command
exec "$@"
