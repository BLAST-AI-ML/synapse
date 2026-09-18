# Overview

Synapse is a modular framework for building digital twin components.
It supports machine and experiment operators with ML-assisted predictions trained on a combination of continuously measured and simulated data.
Synapse embraces emerging [integrated research infrastructures](https://www.nersc.gov/what-we-do/computing-for-science/integrated-research-infrastructure) by deploying a user-facing cloud service, using HPC/cloud compute, and exchanging modular components through container registries.

At the moment, Synapse uses NERSC Spin (control and dashboard), the NERSC Superfacility API (simulation submission and ML training on Perlmutter), and the NERSC container registry.
Synapse is under active development and is being broadened into an AI-accelerated, portable framework.

Synapse enables physicists to couple experimental data, simulations, and ML models trained on both experimental and simulation data.
As an example, the schematic below illustrates how Synapse is used at the Berkeley Lab Laser Accelerator Center (BELLA):

![Synapse overview](synapse_overview.png)

One of the main software components is the graphical user interface (GUI), which is deployed through [Spin](https://docs.nersc.gov/services/spin/) at NERSC.
The application requires access to various data and information sources, as described below.

## Displaying ML predictions

To display ML predictions, the application requires the following:

- **Experiment configuration file**: A YAML file named `config.yaml` stored in the root directory of an experiment's repository that defines the input, output, and calibration variables.
- **Simulation and experimental data points**: Each data point consists of values for the scalar inputs and outputs defined in the experiment configuration file.
  Data points are stored in a [MongoDB](https://www.mongodb.com/) database, with each experiment represented by a separate collection.
  Experimental and simulation data points are stored in the same collection and distinguished by the `experiment_flag` attribute.
- **ML models**: Machine learning models that interpolate between data points and are stored in [MLflow](https://mlflow.org/).
- **Simulation movies** (optional): For certain experiments, users can click on simulation data points to visualize simulation movies.
  The corresponding MP4 files are stored in the Perlmutter shared file system at `/global/cfs/cdirs/m558/superfacility/simulation_data`.
  This directory is mounted on the container image running on Spin.

## Launching ML training at NERSC

ML models can be trained by launching jobs on Perlmutter from the GUI, through the [NERSC Superfacility API](https://docs.nersc.gov/services/sfapi/).
The application requires the following:

- **Superfacility API credential file**: Instructions on generating and uploading the credential file from the GUI are in [Dashboard](dashboard.md).
- **Submission script**: The batch script {repo}`ml/training_pm.sbatch` is copied into the container image pushed to the NERSC registry and deployed through Spin (see {repo}`dashboard.Dockerfile`).
  It serves as a template for Superfacility API job submission when users launch model training from the GUI.
- **Python scripts and configuration files**: These include {repo}`ml/train_model.py`, {repo}`ml/Neural_Net_Classes.py`, and the experiment configuration file `config.yaml`.
  The Python scripts are copied into the ML container image pushed to the NERSC registry (see {repo}`ml.Dockerfile`), and the Superfacility API job runs them from inside that image on Perlmutter, at `/app/ml/`.
  When users launch model training from the GUI, only `config.yaml` is copied to the Perlmutter shared file system, at `/global/cfs/cdirs/m558/superfacility/model_training/`, where the batch job mounts it into the container.
  The `config.yaml` file is automatically populated with the configuration values specified in the GUI before being copied to the shared file system.

## Workflow

The typical workflow is:

1. Add or update an experiment repository under `experiments/synapse-<name>/`.
2. Define `config.yaml` with database, MLflow, input, output, and optional calibration settings.
3. Load experiment and simulation points from MongoDB.
4. Train a model from simulation data, optionally calibrating against experimental data.
5. Register the trained model in MLflow.
6. Use the dashboard to visualize data, query the model, optimize inputs, and launch NERSC jobs.

## External services

Synapse currently assumes these external services:

- MongoDB for experiment and simulation records.
- MLflow for registered model storage.
- NERSC Spin for dashboard deployment.
- NERSC Superfacility API for Perlmutter jobs.
- NERSC container registry for dashboard and ML images.

## Repository layout

- {repo-dir}`dashboard/`: a Trame web application, with its dashboard managers, for exploring experiments, simulations, model predictions, optimization, calibration, and NERSC job controls.
- {repo-dir}`ml/`: the model training script, model classes for Gaussian Process, single Neural Network, and Neural Network ensemble models, and the Perlmutter batch template.
- {repo-dir}`experiments/`: experiment-specific configuration and scripts, usually cloned from private repositories.
- {repo-dir}`tests/`: end-to-end integration checks for the ML pipeline.
- {repo-dir}`docs/`: Sphinx documentation source.

## Copyright notice and license agreement

Synapse is distributed under the `BSD-3-Clause-LBNL` license.
The copyright notice and license agreement are in the {repo}`README <README.md#copyright-notice-and-license-agreement>`, with the full text in {repo}`NOTICE.txt` and {repo}`LICENSE.txt` at the root of the repository.
