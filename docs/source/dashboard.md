# Dashboard

The dashboard is a Trame application rooted at `dashboard/app.py`.
It discovers experiments from `experiments/synapse-*`, reads each experiment's `config.yaml`, connects to MongoDB, loads MLflow models, and builds the GUI used to inspect data and launch jobs.

## Main Managers

- `state_manager.py`: shared Trame server, state, controller, and startup defaults.
- `model_manager.py`: MLflow model lookup, download, evaluation, and model training launch.
- `parameters_manager.py`: input sliders, parameter bounds, and single-simulation launch.
- `outputs_manager.py`: displayed output selection.
- `optimization_manager.py`: model-based input optimization with SciPy.
- `calibration_manager.py`: simulation-to-experiment variable conversion.
- `sfapi_manager.py`: Superfacility API credential upload, Perlmutter status, and job monitoring.
- `error_manager.py`: user-visible error collection.
- `utils.py`: config loading, database access, date filters, and Plotly figures.

## Views

- `/`: experiment selection, plots, parameter controls, optimization, ML controls, calibration controls, and errors.
- `/hpc`: NERSC Superfacility API credential and Perlmutter status panel.
- `/chat`: embedded assistant route for experiment support; currently backed by `https://synapse-chat.lbl.gov/`.

## NERSC Credentials

Simulation and ML training launches require a Superfacility API key file uploaded through the dashboard.
The file must be PEM-formatted and include the Superfacility API client ID as the first line, followed by the private key.
