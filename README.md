# FY24/25 LDRD: IFE Superfacility

This is the project management for our FY24/25 LDRD on creating a blueprint for an IFE superfacility, that can combine experimental and simulation data through ML modeling with the goal to predict good experimental input parameters to achieve a certain goal (e.g., maximize number of particles produced in a certain energy range).

LDRD-funded partners: ATAP AMP (lead), ATAP BELLA (experiments), AMCRD (ML)

Self-funded partners: NERSC, SLAC


## Quick Links

- [Google Drive](https://drive.google.com/drive/u/0/folders/1bwManHU1j67kR008tj7KRuppZPCeaWuj)
- [Meeting Minutes](https://docs.google.com/document/d/1dcpWVORoMVZ1U-bFw1yFQzZOMu8ay4K9hkbavuiiwJI/edit)
- [Deployed Spin app](https://bellasuperfacility.lbl.gov/index.html#/)
- GChat: `Superfacility LDRD`
- Project Management:
  - [Milestones](https://github.com/ECP-WarpX/2024_IFE-superfacility/milestones)
  - [Tasks (Issues)](https://github.com/ECP-WarpX/2024_IFE-superfacility/issues)

## Overview of the deployed workflow

This repo contains scripts and code to show ML predictions from simulations and experimental data, in the BELLA control room, as illustrated schematically here:

![IFE Superfacility Overview](overview_image.png)

One of the main component is the app that runs on NERSC Spin, for which the code is in the `dashboard/` folder. In particular, when running on NERSC Spin, the app needs to access to different sources of information, for different functionalities:

### To display predictions

The app needs the following:

- **Configuration file**: this defines the list of experiments supported by the app, and the input and output quantities for each experiment (located in `dashboard/config`):
- **Simulation and experimental data points**: each data points consists of a set of values for the scalar inputs and scalar outputs defined in the above-mentioned configuration file. These points are stored in a MongoDB database, with each experiment being a separate collection in that database. (Experimental and simulation datapoints are stored in the same collection, and are distinguished the attribute `experimental_flag`.)
- **ML models**: Models that interpolate inbetween datapoints. They are stored in the MongoDB database, in a special collection named `models`.
- **Simulation movies** (optional): for some experiments, the user can click on simulation datapoints and see a movie of the simulation pop up. The corresponding MP4 files are stored on the NERSC shared file system, in the folder `/global/cfs/cdirs/m558/superfacility/simulation_data`. This folder is then mounted on the image that runs on NERSC Spin.

### To launch ML training at NERSC

Retraining of models is done at NERSC using the SFAPI. The app needs the following:

- **SFAPI credential file:** See `dashboard/README` for instructions on how to generate the credential file and upload it into the Spin app.
- **Submission script, Python scripts**: Located in the `ml/` folder ; these files are copied into the Docker image that is pushed to Spin (see `dashboard/Dockerfile`)
