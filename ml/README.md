# ML Training

This guide contains important instructions on how to train ML models within Synapse.

## Prerequisites

Make sure you have installed [conda](https://docs.conda.io/) and [Docker](https://docs.docker.com/).

## Overview

Synapse's ML training is implemented primarily in [train_model.py](train_model.py).
ML models can be trained in two distinct ways:

1. In a local Python environment, for testing and debugging.

2. Through the dashboard (by clicking the ``Train`` button) or through SLURM (by running ``sbatch training_pm.sbatch``).
In both cases, the training runs in a Docker container at NERSC.
This Docker container is pulled from the [NERSC registry](https://registry.nersc.gov) and does not reflect any local changes you may have made to [train_model.py](train_model.py), unless you re-build and re-deploy the container first.

The following sections describe in more detail these two ways of training ML models.

## How to run ML training in a local Python environment

### On a local computer

1. Create the conda environment defined in the lock file (only once):
   ```bash
   conda activate base
   conda install -c conda-forge conda-lock  # if conda-lock is not installed
   conda-lock install --name synapse-ml environment-lock.yml
   ```

2. Open a separate terminal and keep it open while SSH forwarding the database connection:
   ```bash
   ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
   ```

3. Activate the conda environment:
   ```bash
   conda activate synapse-ml
   ```

4. Set up database settings (read-write):
   ```bash
   export SF_DB_ADMIN_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   ```

5. Run the ML training script in test mode:
   ```bash
   python train_model.py --test --model <NN/GP> --config_file <your_config_file>
   ```

### At NERSC

1. Create the conda environment defined in the lock file (only once):
   ```bash
   module load python
   conda env create --prefix /global/cfs/cdirs/m558/$(whoami)/sw/perlmutter/synapse-ml -f environment.yml  # FIXME
   ```

2. Activate the conda environment:
   ```bash
   module load python
   conda activate /global/cfs/cdirs/m558/$(whoami)/sw/perlmutter/synapse-ml
   ```

3. Set up database settings (read-write):
   ```bash
   module load python
   export SF_DB_ADMIN_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   ```

4. Run the ML training script in test mode:
   ```bash
   python train_model.py --test --model <NN/GP> --config_file <your_config_file>
   ```

## Training through the dashboard or through SLURM

> [!WARNING]
> Pushing a new Docker container affects training jobs launched from your locally-deployed dashboard, but also from the production dashboard (deployed at NERSC through Spin), because in both cases the ML training runs in a Docker container at NERSC, which is pulled from the [NERSC registry](https://registry.nersc.gov).
> Currently, this is the only way to test the end-to-end integration of the dashboard with the ML training workflow.

1. Move to the root directory of the repository.

2. Build the Docker image based on `Dockerfile`:
   ```bash
   docker build --platform linux/amd64 -t synapse-ml -f ml.Dockerfile .
   ```

3. (Optional) As you develop the container, you might want to prune old, unused images periodically in order to free space on your development machine:
   ```bash
   docker system prune -a
   ```

4. Publish the container privately to [NERSC registry](https://registry.nersc.gov):
   ```bash
   docker login registry.nersc.gov
   # Username: your NERSC username
   # Password: your NERSC password without 2FA
   ```
   ```bash
   docker tag synapse-ml:latest registry.nersc.gov/m558/superfacility/synapse-ml:latest
   docker tag synapse-ml:latest registry.nersc.gov/m558/superfacility/synapse-ml:$(date "+%y.%m")
   docker push -a registry.nersc.gov/m558/superfacility/synapse-ml
   ```
    This has been also automated through the Python script [publish_container.py](../publish_container.py), which can be executed via
    ```bash
    python publish_container.py --ml
    ```

5. (Optional) Run the Docker container manually on Perlmutter:
   ```bash
   ssh perlmutter-p1.nersc.gov

   podman-hpc login --username $USER registry.nersc.gov
   # Password: your NERSC password without 2FA

   podman-hpc pull registry.nersc.gov/m558/superfacility/synapse-ml:latest
   ```

   Ensure the file `$HOME/db.profile` contains a line `export SF_DB_ADMIN_PASSWORD=...` with the read-write password to the database.

   ```bash
   salloc -N 1 --ntasks-per-node=1 -t 1:00:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 1 -A m558

   podman-hpc run --gpu -v /etc/localtime:/etc/localtime -v $HOME/db.profile:/root/db.profile -v /path/to/config.yaml:/app/ml/config.yaml --rm -it registry.nersc.gov/m558/superfacility/synapse-ml:latest python -u /app/ml/train_model.py --test --config_file /app/ml/config.yaml --model NN
   ```
   Note that `-v /etc/localtime:/etc/localtime` is necessary to synchronize the time zone in the container with the host machine.


> [!NOTE]
> When we run ML training jobs through the dashboard, we use NERSC's Superfacility API with the collaboration account `sf558`.
> Since this is a non-interactive, non-user account, we also use a custom user to pull the image from the [NERSC registry](https://registry.nersc.gov) to Perlmutter.
> The registry login credentials need to be prepared (only once) in the `$HOME` of user `sf558` (`/global/homes/s/sf558/`), in a file named `registry.profile` with the following content:
> ```bash
> export REGISTRY_USER="robot\$m558+perlmutter-nersc-gov"
> export REGISTRY_PASSWORD="..."
> ```

## References

* [Podman at NERSC](https://docs.nersc.gov/development/containers/podman-hpc/overview/)
* [Using NERSC's `registry.nersc.gov`](https://docs.nersc.gov/development/containers/registry/)
