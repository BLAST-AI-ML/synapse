# Simulations

Synapse treats simulation support as experiment-owned code.
The dashboard only needs to know where to find scripts and how to submit a job through the NERSC Superfacility API.

## Simulation script layout

For dashboard-triggered single simulations, an experiment may provide:

```text
experiments/synapse-<experiment>/simulation_scripts/
  submission_script_single
  templates/
    ...
```

The dashboard enables the `Simulate` button only when all three of the following hold: `submission_script_single` exists, Perlmutter reports status `active`, and no dashboard-launched simulation is already running.
Before submission, it writes the current dashboard parameters to `single_simulation_parameters.yaml` after converting experimental variables to simulation variables.

## Submission flow

1. User uploads valid Superfacility API credentials.
2. Dashboard checks Perlmutter status.
3. User clicks `Simulate`.
4. Files from `simulation_scripts/templates/` and the generated parameter YAML are uploaded to:

   ```text
   /global/cfs/cdirs/m558/superfacility/simulation_running/<experiment>/templates
   ```

5. The dashboard reads `submission_script_single` and submits it through Superfacility API.
6. Job status is polled until a terminal state, such as completed, failed, or cancelled.

## Parameter scans

Some experiment repositories also include `submission_script_multi` or custom scan scripts.
These scripts are experiment-specific and are usually run manually on Perlmutter.

## Simulation outputs

Simulation records should be written to the experiment's MongoDB collection with `experiment_flag: 0`.
Field names should match either the experiment config outputs or the configured simulation calibration variable names.

When a simulation record includes a `data_directory` under `/global/cfs/cdirs/m558/superfacility/simulation_data`, the dashboard can link the record to a plot file in that directory's `plots/` subdirectory.
It prefers a single MP4 file and otherwise falls back to the last PNG file whose name contains `iteration`.
This support is optional because the experiment's simulation scripts must create the record and its corresponding files.
