# Table of Contents
* [Overview](#Overview)
* [Run the Dashboard Locally](#Run-the-Dashboard-Locally)
    * [Without Docker](#Without-Docker)
    * [With Docker](#With-Docker)
* [Run the Dashboard at NERSC](#Run-the-Dashboard-at-NERSC)
* [Get the Superfacility API Credentials](#Get-the-Superfacility-API-Credentials)
* [For Maintainers](#For-Maintainers)
    * [Generate the conda environment lock file](#Generate-the-conda-environment-lock-file)
    * [Build and push the Docker container to NERSC](#Build-and-push-the-Docker-container-to-NERSC)
* [References](#References)

# Overview

The Synapse dashboard provides a web interface for working with data from experiments, simulations, and ML models.

The dashboard can be run in two distinct ways:

1. Locally on your computer.

2. At NERSC through Spin.

# Run the Dashboard Locally

This section describes how to develop and use the dashboard locally.

## Without Docker

### Prepare the conda environment

1. Move to the [dashboard/](./) directory.

2. Activate the conda environment `base`:
```bash
conda activate base
```

3. Install `conda-lock` if not installed yet:
```bash
conda install -c conda-forge conda-lock
```

4. Create the conda environment `synapse-gui`:
```bash
conda-lock install --name synapse-gui environment-lock.yml
```

### Run the dashboard

1. Create an SSH tunnel to access the MongoDB database at NERSC (in a separate terminal):
   ```bash
   ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
   ```

2. Move to the [dashboard/](./) directory.

3. Set up the database settings (read-only) and, if using AmSC MLflow, the API key:
   ```bash
   export SF_DB_HOST='127.0.0.1'
   export SF_DB_READONLY_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   export AM_SC_API_KEY='your_amsc_api_key_here'        # Required when MLflow tracking_uri is AmSC (e.g. https://mlflow.american-science-cloud.org)
   ```

4. Activate the conda environment `synapse-gui`:
   ```bash
   conda activate synapse-gui
   ```

5. Run the dashboard as a web application:
   ```bash
   python -u app.py --port 8080
   ```

## With Docker

### Run the dashboard

1. Create an SSH tunnel to access the MongoDB database at NERSC (in a separate terminal):
   ```bash
   ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
   ```

2. Move to the root directory of the repository.

3. Build the Docker image as described [below](#build-the-docker-image).

4. Run the Docker container:
   ```bash
   docker run --network=host -v /etc/localtime:/etc/localtime -v $PWD/ml:/app/ml -e SF_DB_HOST='127.0.0.1' -e SF_DB_READONLY_PASSWORD='your_password_here' -e AM_SC_API_KEY='your_amsc_api_key_here' synapse-gui
   ```
   For debugging, you can enter the container without starting the app:
   ```bash
   docker run --network=host -v /etc/localtime:/etc/localtime -v $PWD/ml:/app/ml -e SF_DB_HOST='127.0.0.1' -e SF_DB_READONLY_PASSWORD='your_password_here' -e AM_SC_API_KEY='your_amsc_api_key_here' -it synapse-gui bash
   ```
   Note that `-v /etc/localtime:/etc/localtime` is necessary to synchronize the time zone in the container with the host machine.

# Run the Dashboard at NERSC

Connect to the [dashboard](https://bellasuperfacility.lbl.gov/) deployed at NERSC through Spin and play around!
Remember that you need to upload valid Superfacility API credentials in order to launch simulations or train ML models directly from the dashboard.

# Get the Superfacility API Credentials

Following the instructions at [docs.nersc.gov/services/sfapi/authentication/#client](https://docs.nersc.gov/services/sfapi/authentication/#client):

1. Log in to your profile page at [iris.nersc.gov/profile](https://iris.nersc.gov/profile).

2. Click the icon with your username in the upper right of the profile page.

3. Scroll down to the section "Superfacility API Clients" and click "New Client".

4. Enter a client name (e.g., "Synapse"), choose `sf558` for the user, choose "Red" security level, and select either "Your IP" or "Spin" from the "IP Presets" menu, depending on whether the key will be used from a local computer or from Spin.

5. Download the private key file (in pem format) and save it as `priv_key.pem` in the root directory of the dashboard.
   Each time the dashboard is launched, it will automatically find the existing key file and load the corresponding credentials.

6. Copy your client ID and add it on the first line of your private key file as described in the instructions at [nersc.github.io/sfapi_client/quickstart/#storing-keys-in-files](https://nersc.github.io/sfapi_client/quickstart/#storing-keys-in-files):
   ```
   randmstrgz
   -----BEGIN RSA PRIVATE KEY-----
   ...
   -----END RSA PRIVATE KEY-----
   ```

7. Run `chmod 600 priv_key.pem` to change the permissions of your private key file to read/write only.

# For Maintainers

## Generate the conda environment lock file

1. Move to the directory [dashboard/](.).

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
> Pushing a new Docker container affects the production dashboard deployed at NERSC through Spin.

> [!TIP]
> Run this workflow automatically with the Python script [publish_container.py](../publish_container.py):
> ```bash
> python publish_container.py --gui
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
   docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-gui -f dashboard.Dockerfile .
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
   docker tag synapse-gui:latest registry.nersc.gov/m558/superfacility/synapse-gui:latest
   docker tag synapse-gui:latest registry.nersc.gov/m558/superfacility/synapse-gui:$(date "+%y.%m")
   ```

4. Push the Docker container:
   ```bash
   docker push -a registry.nersc.gov/m558/superfacility/synapse-gui
   ```

# References

* [Using NERSC's `registry.nersc.gov`](https://docs.nersc.gov/development/containers/registry/)
* [Superfacility API authentication](https://docs.nersc.gov/services/sfapi/authentication/#client)
