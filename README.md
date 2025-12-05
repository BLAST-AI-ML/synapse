# Synergistic Software Platform for AI, Physics Simulations, and Experiments (Synapse)

## Overview

Synapse is a software platform that enables experimental physicists to couple experimental data, simulations, and machine learning (ML) models trained on experimental and simulation data. The schematic below illustrates the architecture for the Berkeley Lab Laser Accelerator Center (BELLA):

![Synapse overview](synapse_overview.png)

One of the main software components is the graphical user interface (GUI), which is deployed via [Spin](https://docs.nersc.gov/services/spin/) at NERSC.
The application requires access to various data and information sources, as described below.

### Displaying ML predictions

To display ML predictions, the application requires the following:

- **Experiment configuration file**: A YAML file named `config.yaml` stored in the root directory of an experiment's repository that defines the input, output, and calibration variables.
- **Simulation and experimental data points**: Each data point consists of values for the scalar inputs and outputs defined in the experiment configuration file.
Data points are stored in a [MongoDB](https://www.mongodb.com/) database, where each experiment is represented by a separate collection.
Experimental and simulation data points are stored in the same collection and are distinguished by the `experimental_flag` attribute.
- **ML models**: Machine learning models that interpolate between data points and are stored in a separate MongoDB collection named `models`.
- **Simulation movies** (optional): For certain experiments, users can click on simulation data points to visualize simulation movies.
The corresponding MP4 files are stored in the Perlmutter shared file system at `/global/cfs/cdirs/m558/superfacility/simulation_data`.
This directory is mounted on the container image running on Spin.

### Launching ML training at NERSC

The application requires the following:

- **Superfacility API credential file**: Instructions on generating and uploading the credential file from the GUI are in [dashboard/README.md](dashboard/README.md).
- **Submission script**: The batch script [ml/training_pm.sbatch](ml/training_pm.sbatch) is copied into the container image pushed to the NERSC registry and deployed via Spin (see [dashboard.Dockerfile](dashboard.Dockerfile)). It serves as a template for Superfacility API job submission when users launch model training from the GUI.
- **Python scripts and configuration files**: These include [ml/train_model.py](ml/train_model.py), [ml/Neural_Net_Classes.py](ml/Neural_Net_Classes.py), and the experiment configuration file `config.yaml`.
They are copied into the container image pushed to the NERSC registry and deployed via Spin (see [dashboard.Dockerfile](dashboard.Dockerfile)).
When users launch model training from the GUI, these files are copied to the Perlmutter shared file system at `/global/cfs/cdirs/m558/superfacility/model_training/src/` for access by the Superfacility API job.

## Copyright Notice and License Agreement

Synapse v1.0 Copyright (c) 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

Please find the full copyright notice in [NOTICE.txt](NOTICE.txt) and the full license agreement in [LICENSE.txt](LICENSE.txt).

The SPDX license identifier is `BSD-3-Clause-LBNL`.
