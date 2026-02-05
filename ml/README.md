# Table of Contents
* [Overview](#Overview)
* [Train ML Models Locally](#Train-ML-Models-Locally)
    * [Without Docker](#Without-Docker)
    * [With Docker](#With-Docker)
* [Train ML Models at NERSC](#Train-ML-Models-at-NERSC)
    * [Manually without Docker](#Manually-without-Docker)
    * [Manually with Docker](#Manually-with-Docker)
    * [Through the dashboard](#Through-the-dashboard)
* [For Maintainers](#For-Maintainers)
    * [Generate the conda environment lock file](#Generate-the-conda-environment-lock-file)
    * [Build and push the Docker container to NERSC](#Build-and-push-the-Docker-container-to-NERSC)
* [References](#References)

# Overview

Synapse's ML training is implemented primarily in [train_model.py](train_model.py).

ML models can be trained in two distinct ways:

1. Locally on your computer.

2. At NERSC, either manually or through the dashboard.

# Train ML Models Locally

This section describes how to train ML models locally.

## Without Docker

### Prepare the conda environment

1. Move to the [ml/](./) directory.

2. Activate the conda environment `base`:
   ```bash
   conda activate base
   ```

3. Install `conda-lock` if not installed yet:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Create the conda environment `synapse-ml`:
   ```bash
   conda-lock install --name synapse-ml environment-lock.yml
   ```

### Run the training

1. Create an SSH tunnel to access the MongoDB database at NERSC (in a separate terminal):
   ```bash
   ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
   ```

2. Move to the [ml/](./) directory.

3. Set up the database settings (read-write):
   ```bash
   export SF_DB_ADMIN_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   ```

4. Activate the conda environment `synapse-ml`:
   ```bash
   conda activate synapse-ml
   ```

5. Run the ML training script in test mode:
   ```bash
   python train_model.py --test --model <your_model> --config_file <your_config_file>
   ```

## With Docker

Coming soon.

# Train ML Models at NERSC

This section describes how to train ML models at NERSC.

## Manually without Docker

### Prepare the conda environment

1. Move to the [ml/](./) directory.

2. Activate your own user base conda environment:
   ```bash
   module load python
   conda activate <your_base_env>
   ```

3. Install `conda-lock` if not installed yet:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Create the conda environment `synapse-ml`:
   ```bash
   conda-lock install --name synapse-ml environment-lock.yml
   ```

### Run the training

1. Move to the [ml/](./) directory.

2. Set up the database settings (read-write):
   ```bash
   export SF_DB_ADMIN_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   ```

3. Activate the conda environment `synapse-ml`:
   ```bash
   module load python
   conda activate synapse-ml
   ```

4. Run the ML training script in test mode:
   ```bash
   python train_model.py --test --model <your_model> --config_file <your_config_file>
   ```

## Manually with Docker

> [!WARNING]
> Note that the Docker container is pulled from the [NERSC registry](https://registry.nersc.gov) and does not reflect any local changes you may have made to [train_model.py](train_model.py), unless you re-build and re-deploy the container first.

1. Log in to Perlmutter:
   ```bash
   ssh perlmutter-p1.nersc.gov
   ```

2. Ensure the file `$HOME/db.profile` contains the line `export SF_DB_ADMIN_PASSWORD='your_password_here'` with the read-write password to the database.

3. Pull the Docker container:
   ```bash
   podman-hpc login --username $USER registry.nersc.gov
   # Password: your NERSC password without 2FA
   podman-hpc pull registry.nersc.gov/m558/superfacility/synapse-ml:latest
   ```

4. Allocate a GPU node and run the container:
   ```bash
   salloc -N 1 --ntasks-per-node=1 -t 1:00:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 1 -A m558
   podman-hpc run --gpu -v /etc/localtime:/etc/localtime -v $HOME/db.profile:/root/db.profile -v /path/to/config.yaml:/app/ml/config.yaml --rm -it registry.nersc.gov/m558/superfacility/synapse-ml:latest python -u /app/ml/train_model.py --test --config_file /app/ml/config.yaml --model NN
   ```
   Note that `-v /etc/localtime:/etc/localtime` is necessary to synchronize the time zone in the container with the host machine.

## Through the dashboard

> [!WARNING]
> When we train ML models through the dashboard, we use NERSC's Superfacility API with the collaboration account `sf558`.
> Since this is a non-interactive, non-user account, we also use a custom user to pull the image from the [NERSC registry](https://registry.nersc.gov) to Perlmutter.
> The registry login credentials need to be prepared (only once) in the `$HOME` of user `sf558` (`/global/homes/s/sf558/`), in a file named `registry.profile` with the following content:
> ```bash
> export REGISTRY_USER="robot\$m558+perlmutter-nersc-gov"
> export REGISTRY_PASSWORD="..."
> ```

Connect to the [dashboard](https://bellasuperfacility.lbl.gov/) deployed at NERSC through Spin and click the `Train` button in the `ML` panel.
Remember that you need to upload valid Superfacility API credentials in order to launch simulations or train ML models directly from the dashboard.

# For Maintainers

## Generate the conda environment lock file

1. Move to the directory [ml/](./).

2. Activate the conda environment `base`:
   ```bash
   conda activate base
   ```

3. Install `conda-lock` if not installed yet:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Generate the conda environment lock file:
   ```bash
   conda-lock --file environment.yml --lockfile environment-lock.yml
   ```

## Build and push the Docker container to NERSC

> [!WARNING]
> Pushing a new Docker container affects both the ML training jobs launched from a dashboard deployed locally and the ML training jobs launched from the dashboard deployed at NERSC, because in both cases the ML training runs in a Docker container pulled from the [NERSC registry](https://registry.nersc.gov).
> Currently, this is the only way to test the end-to-end integration of the dashboard with the ML training workflow.

> [!TIP]
> Run this workflow automatically with the Python script [publish_container.py](../publish_container.py):
> ```bash
> python publish_container.py --ml
> ```

> [!TIP]
> Prune old, unused images periodically in order to free up space on your machine:
> ```bash
> docker system prune -a
> ```

### Build the Docker image

1. Move to the root directory of the repository.

2. Build the Docker image:
   ```bash
   docker build --platform linux/amd64 -t synapse-ml -f ml.Dockerfile .
   ```

### Push the Docker container

1. Move to the root directory of the repository.

2. Login to the [NERSC registry](https://registry.nersc.gov):
   ```bash
   docker login registry.nersc.gov
   # Username: your NERSC username
   # Password: your NERSC password without 2FA
   ```

3. Tag the Docker image:
   ```bash
   docker tag synapse-ml:latest registry.nersc.gov/m558/superfacility/synapse-ml:latest
   docker tag synapse-ml:latest registry.nersc.gov/m558/superfacility/synapse-ml:$(date "+%y.%m")
   ```

4. Push the Docker container:
   ```bash
   docker push -a registry.nersc.gov/m558/superfacility/synapse-ml
   ```

# References

* [Using NERSC's `registry.nersc.gov`](https://docs.nersc.gov/development/containers/registry/)
* [Podman at NERSC](https://docs.nersc.gov/development/containers/podman-hpc/overview/)
