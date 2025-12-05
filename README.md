# Synergistic Software Platform for AI, Physics Simulations, and Experiments (Synapse)

## Overview

Synapse is a software platform that enables experimental physicists in a control room to couple experimental data, simulations, and machine learning (ML) models trained on experimental data and simulations, as illustrated schematically here for the case of the Berkeley Lab Laser Accelerator Center (BELLA):

![Synapse overview](synapse_overview.png)

One of the main software components is the graphical user interface (GUI) deployed via [Spin](https://docs.nersc.gov/services/spin/) at NERSC.
The application requires access to various data and information sources, as described in the following sections.

### To display ML predictions

The application needs the following:

- **Experiment configuration file**: A YAML file named `config.yaml` stored in the root directory of an experiment's repository.
It defines the input, output, and calibration variables for the experiment.
- **Simulation and experimental data points**: Each data point consists of values for the scalar inputs and outputs defined in the experiment configuration file.
Data points are stored in a [MongoDB](https://www.mongodb.com/) database, where each experiment is represented by a separate collection.
Experimental and simulation data points are stored in the same collection and are distinguished by the `experimental_flag` attribute.
- **ML models**: Machine learning models that interpolate between data points.
They are stored in a separate MongoDB collection named `models`.
- **Simulation movies** (optional): For certain experiments, users can click on simulation data points to visualize simulation movies.
The corresponding MP4 files are stored in the NERSC shared file system at `/global/cfs/cdirs/m558/superfacility/simulation_data`.
This directory is mounted on the container image running on Spin.

### To launch ML training at NERSC

ML models training can be submitted as a NERSC job from the GUI via the [NERSC Superfacility API](https://docs.nersc.gov/services/sfapi/).
The application needs the following:

- **Superfacility API credential file**: See [dashboard/README.md](dashboard/README.md) for instructions on generating and uploading the credential file from the GUI.
- **Submission script**: See [ml/training_pm.sbatch](ml/training_pm.sbatch).
This file is copied into the container image that is pushed to the NERSC registry and deployed via Spin (see [dashboard/Dockerfile](dashboard/Dockerfile)). It serves as a template for Superfacility API job submission when users launch model training from the GUI.
- **Python scripts and configuration files**: See [ml/train_model.py](ml/train_model.py), [ml/Neural_Net_Classes.py](ml/Neural_Net_Classes.py), and [dashboard/config/variables.yml](dashboard/config/variables.yml).
These files are copied into the container image that is pushed to the NERSC registry and deployed via Spin (see [dashboard/Dockerfile](dashboard/Dockerfile)).
When users launch model training from the GUI, these files are copied to the Perlmutter shared file system at `/global/cfs/cdirs/m558/superfacility/model_training/src/` so that the Superfacility API job can access them.

## Copyright Notice and License Agreement

Synapse v1.0 Copyright (c) 2025, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

Please find the full copyright notice in [NOTICE.txt](NOTICE.txt) and the full license agreement in [LICENSE.txt](LICENSE.txt).

The SPDX license identifier is `BSD-3-Clause-LBNL`.
