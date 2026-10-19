# Data model

MongoDB stores experiment and simulation records.
The collection name should match the `experiment` value in `config.yaml`.

## Record types

- `experiment_flag: 1`: experimental data.
- `experiment_flag: 0`: simulation data.

## Required fields

Each record must contain fields for the configured input and output variables, with names that match `config.yaml`.

Simulation records may use simulation-space names when `simulation_calibration` maps those names back to experimental variables.

## Optional fields

The dashboard uses these when present:

- `date`: filtering and hover text for experimental records.
- `scan_number`: hover text.
- `shot_number`: hover text.
- `_id`: hover text, and the key the dashboard uses to fetch the clicked record, whose `data_directory` then locates the linked simulation media, such as MP4 files described in [Simulation outputs](simulations.md#simulation-outputs).

## Date filtering

Dashboard date filtering applies only to experimental records.
Simulation records are loaded without the date filter.
