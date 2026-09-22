# Dashboard

The Synapse dashboard provides a web interface for working with experiment data, simulation data, and ML models.

The dashboard can be run in two distinct ways:

1. Locally on your computer.

2. At NERSC through Spin.

The dashboard is a Trame application rooted in {repo}`dashboard/app.py`.
It discovers experiments from the subdirectories of {repo-dir}`experiments/`, stripping the `synapse-` prefix from each directory name, reads each experiment's `config.yaml`, connects to MongoDB, loads MLflow models, and builds the GUI used to inspect data and launch jobs.

## Run the dashboard locally

This section describes how to develop and use the dashboard locally.

### Without Docker

#### Prepare the conda environment

1. Move to the {repo-dir}`dashboard/` directory.

2. Activate the conda environment `base`:
```bash
conda activate base
```

3. Install `conda-lock` if it is not already installed:
```bash
conda install -c conda-forge conda-lock
```

4. Create the conda environment `synapse-gui` from {repo}`environment-lock.yml <dashboard/environment-lock.yml>`:
```bash
conda-lock install --name synapse-gui environment-lock.yml
```

#### Run the dashboard

1. If the computer running the dashboard cannot directly reach the host specified by `database.host`, create an SSH tunnel to the MongoDB database through a gateway node in a separate terminal, using `database.host` and `database.port` from your experiment's `config.yaml`:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   ssh -L <database.port>:<database.host>:<database.port> <username>@<gateway_host> -N
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
   ```
   :::
   ::::

2. If you created the SSH tunnel, set `database.host` to `127.0.0.1` in your local copy of `config.yaml`, so that the dashboard connects through it, but do not commit this change.
   ```yaml
   database:
     host: "127.0.0.1"
   ```

3. Move to the {repo-dir}`dashboard/` directory.

4. Set up the read-only database password and the MLflow API key:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   export <database.password_ro_env>='your_password_here'  # Use SINGLE quotes around the password!
   export <mlflow.api_key_env>='your_api_key_here'         # Required when MLflow tracking_uri is AmSC
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   export SF_DB_READONLY_PASSWORD='your_password_here'  # Use SINGLE quotes around the password!
   export AM_SC_API_KEY='your_amsc_api_key_here'        # Required when MLflow tracking_uri is AmSC
   ```
   :::
   ::::

5. Activate the conda environment `synapse-gui`:
   ```bash
   conda activate synapse-gui
   ```

6. Run the dashboard as a web application:
   ```bash
   python -u app.py --port 8080
   ```

### With Docker

#### Run the container

1. If the computer running the dashboard cannot directly reach the host specified by `database.host`, create an SSH tunnel to the MongoDB database through a gateway node in a separate terminal, using `database.host` and `database.port` from your experiment's `config.yaml`:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   ssh -L <database.port>:<database.host>:<database.port> <username>@<gateway_host> -N
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   ssh -L 27017:mongodb05.nersc.gov:27017 <username>@dtn03.nersc.gov -N
   ```
   :::
   ::::

2. If you created the SSH tunnel, set `database.host` to `127.0.0.1` in your local copy of `config.yaml`, but do not commit this change.
   Do this before building the image, because {repo}`dashboard.Dockerfile` copies {repo-dir}`experiments/` into it.

3. Move to the root directory of the repository.

4. Build the Docker image as described [below](#build-the-docker-image).

5. Run the Docker container with the read-only database password and the MLflow API key:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   docker run --network=host -v /etc/localtime:/etc/localtime -v $PWD/ml:/app/ml -e <database.password_ro_env>='your_password_here' -e <mlflow.api_key_env>='your_api_key_here' synapse-gui
   ```
   For debugging, you can enter the container without starting the app:
   ```bash
   docker run --network=host -v /etc/localtime:/etc/localtime -v $PWD/ml:/app/ml -e <database.password_ro_env>='your_password_here' -e <mlflow.api_key_env>='your_api_key_here' -it synapse-gui bash
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   docker run --network=host -v /etc/localtime:/etc/localtime -v $PWD/ml:/app/ml -e SF_DB_READONLY_PASSWORD='your_password_here' -e AM_SC_API_KEY='your_amsc_api_key_here' synapse-gui
   ```
   For debugging, you can enter the container without starting the app:
   ```bash
   docker run --network=host -v /etc/localtime:/etc/localtime -v $PWD/ml:/app/ml -e SF_DB_READONLY_PASSWORD='your_password_here' -e AM_SC_API_KEY='your_amsc_api_key_here' -it synapse-gui bash
   ```
   :::
   ::::

   Note that `-v /etc/localtime:/etc/localtime` is necessary to synchronize the time zone in the container with the host machine.

## Run the dashboard at NERSC

Connect to the dashboard deployed for your project at NERSC through Spin and explore it:

::::{tab-set}
:sync-group: deployment

:::{tab-item} General
:sync: general

Open `https://<dashboard_url>/`, the URL of your project's Spin deployment.
:::

:::{tab-item} Project Example: BELLA @ NERSC
:sync: bella-nersc

Open [bellasuperfacility.lbl.gov](https://bellasuperfacility.lbl.gov/).
:::
::::

You need to upload valid Superfacility API credentials before you can launch simulations or train ML models directly from the dashboard.

## Generate Superfacility API credentials

Follow the instructions at [docs.nersc.gov/services/sfapi/authentication/#client](https://docs.nersc.gov/services/sfapi/authentication/#client):

1. Log in to your profile page at [iris.nersc.gov/profile](https://iris.nersc.gov/profile).

2. Click the icon with your username in the upper right of the profile page.

3. Scroll down to the section "Superfacility API Clients" and click "New Client".

4. Enter a client name (e.g., "Synapse"), choose the user that runs the jobs launched from the dashboard, choose "Red" security level, and select either "Your IP" or "Spin" from the "IP Presets" menu, depending on whether the key will be used from a local computer or from Spin.
   ML training jobs expect credential files in the `$HOME` of that user, as described in [Through the dashboard](ml-training.md#through-the-dashboard).

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   Choose your own NERSC user or a collaboration account of your NERSC project.
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   Choose the collaboration account `sf558`.
   :::
   ::::

5. Download the private key file in PEM format and save it as `priv_key.pem` in the root directory of the dashboard.
   Each time the dashboard is launched, it will automatically find the existing key file and load the corresponding credentials.

6. Copy your client ID and add it on the first line of your private key file as described in the instructions at [nersc.github.io/sfapi_client/quickstart/#storing-keys-in-files](https://nersc.github.io/sfapi_client/quickstart/#storing-keys-in-files):
   ```
   randmstrgz
   -----BEGIN RSA PRIVATE KEY-----
   ...
   -----END RSA PRIVATE KEY-----
   ```

7. Run `chmod 600 priv_key.pem` to restrict your private key file to read/write access only.

## Manager modules

- {repo}`state_manager.py <dashboard/state_manager.py>`: shared Trame server, state, controller, and startup defaults.
- {repo}`model_manager.py <dashboard/model_manager.py>`: MLflow model lookup, download, evaluation, and model training launch.
- {repo}`parameters_manager.py <dashboard/parameters_manager.py>`: input sliders, parameter bounds, and single-simulation launch.
- {repo}`outputs_manager.py <dashboard/outputs_manager.py>`: displayed output selection.
- {repo}`optimization_manager.py <dashboard/optimization_manager.py>`: model-based input optimization with SciPy.
- {repo}`calibration_manager.py <dashboard/calibration_manager.py>`: conversion between simulation and experiment variables, in both directions.
- {repo}`sfapi_manager.py <dashboard/sfapi_manager.py>`: Superfacility API credential upload, Perlmutter status, and job monitoring.
- {repo}`error_manager.py <dashboard/error_manager.py>`: user-visible error collection.
- {repo}`utils.py <dashboard/utils.py>`: config loading, database access, date filters, and Plotly figures.

## Routes

The dashboard has three routes, reachable from the navigation drawer:

- `/` ("Digital Twin Prototype"): the plots card, next to a tab group with three tabs.
  The `Parameters` tab holds the displayed output selector, the input parameter controls, and the plot depth control.
  The `Optimization` tab holds the optimization controls.
  The `ML` tab holds the model controls and the calibration controls.
- `/hpc` ("HPC Connection"): Genesis AmSC IRI API or NERSC Superfacility API credential and HPC status panel.
- `/chat` ("AI Assistant"): embedded assistant for experiment support.
  It loads [synapse-chat.lbl.gov](https://synapse-chat.lbl.gov/), which is hardcoded in {repo}`dashboard/app.py`.

The experiment selector, the date range selector, and the error panel belong to the shared layout rather than to any single route, so they appear on all three.

## Credential file format

Simulation and ML training launches require a Superfacility API key file uploaded through the dashboard.
The file must be PEM-formatted and include the Superfacility API client ID as the first line, followed by the private key.

## For maintainers

### Generate the conda environment lock file

1. Move to the directory {repo-dir}`dashboard/`.

2. Activate the conda environment `base`:
   ```bash
   conda activate base
   ```

3. Install `conda-lock` if it is not already installed:
   ```bash
   conda install -c conda-forge conda-lock
   ```

4. Generate the conda environment lock file from {repo}`environment.yml <dashboard/environment.yml>`:
   ```bash
   conda-lock --file environment.yml --lockfile environment-lock.yml
   ```

### Build and push the Docker image to NERSC

```{warning}
Pushing a new Docker image affects the production dashboard deployed through Spin at NERSC.
```

````{tip}
Run this workflow automatically with the Python script {repo}`publish_container.py`:
```bash
python publish_container.py --gui
```
The script pushes to `registry.nersc.gov/m558/superfacility`, which is hardcoded for the BELLA deployment.
For other projects, follow the steps below.
````

````{tip}
Prune old, unused images periodically to free up space on your machine:
```bash
docker system prune -a
```
````

#### Build the Docker image

1. Move to the root directory of the repository.

2. Build the Docker image defined in {repo}`dashboard.Dockerfile`:
   ```bash
   docker build --platform linux/amd64 --output type=image,oci-mediatypes=true -t synapse-gui -f dashboard.Dockerfile .
   ```

#### Push the Docker image

1. Move to the root directory of the repository.

2. Log in to the [NERSC registry](https://registry.nersc.gov):
   ```bash
   docker login registry.nersc.gov
   # Username: your NERSC username
   # Password: your NERSC password without 2FA
   ```

3. Tag the Docker image with the registry path of your NERSC project:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   docker tag synapse-gui:latest registry.nersc.gov/<nersc_project>/[<namespace>/]synapse-gui:latest
   docker tag synapse-gui:latest registry.nersc.gov/<nersc_project>/[<namespace>/]synapse-gui:$(date "+%y.%m")
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   docker tag synapse-gui:latest registry.nersc.gov/m558/superfacility/synapse-gui:latest
   docker tag synapse-gui:latest registry.nersc.gov/m558/superfacility/synapse-gui:$(date "+%y.%m")
   ```
   :::
   ::::

4. Push the Docker image:

   ::::{tab-set}
   :sync-group: deployment

   :::{tab-item} General
   :sync: general

   ```bash
   docker push -a registry.nersc.gov/<nersc_project>/[<namespace>/]synapse-gui
   ```
   :::

   :::{tab-item} Project Example: BELLA @ NERSC
   :sync: bella-nersc

   ```bash
   docker push -a registry.nersc.gov/m558/superfacility/synapse-gui
   ```
   :::
   ::::

## References

* [Using NERSC's `registry.nersc.gov`](https://docs.nersc.gov/development/containers/registry/)
* [Superfacility API authentication](https://docs.nersc.gov/services/sfapi/authentication/#client)
