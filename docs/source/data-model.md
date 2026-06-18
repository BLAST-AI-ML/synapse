# Data Model

MongoDB stores experiment and simulation records.
The collection name should match `experiment` in `config.yaml`.

## Record Types

- `experiment_flag: 1`: experimental data.
- `experiment_flag: 0`: simulation data.

## Required Fields

Records should include fields for the configured input and output variable names.

Simulation records may use simulation-space names when `simulation_calibration` maps them back to experimental names.

## Optional Fields

The dashboard uses these when present:

- `date`: filtering and hover text for experimental records.
- `scan_number`: hover text.
- `shot_number`: hover text.
- `_id`: hover text and simulation media lookup.

## Date Filtering

Dashboard date filtering applies only to experimental records.
Simulation records are loaded without the date filter.
