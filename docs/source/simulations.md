# Simulations

Synapse treats simulation support as experiment-owned code.
The dashboard only needs to know where to find scripts and how to submit a job through the NERSC Superfacility API.

## Directory Layout

For dashboard-triggered single simulations, an experiment may provide:

```text
experiments/synapse-<experiment>/simulation_scripts/
  submission_script_single
  templates/
    ...
```

If `submission_script_single` exists, the dashboard enables the `Simulate` button.
Before submission, it writes the current dashboard parameters to `single_simulation_parameters.yaml` after converting experimental variables to simulation variables.

## Submission Flow

1. User uploads valid Superfacility API credentials.
2. Dashboard checks Perlmutter status.
3. User clicks `Simulate`.
4. Files from `simulation_scripts/templates/` and the generated parameter YAML are uploaded to:

   ```text
   /global/cfs/cdirs/m558/superfacility/simulation_running/<experiment>/templates
   ```

5. The dashboard reads `submission_script_single` and submits it through Superfacility API.
6. Job status is polled until a terminal state, such as completed, failed, or cancelled.

## Parameter Scans

Some experiment repositories also include `submission_script_multi` or custom scan scripts.
These scripts are experiment-specific and are usually run manually on Perlmutter.

## Simulation Outputs

Simulation records should be written to the experiment's MongoDB collection with `experiment_flag: 0`.
Field names should match either the experiment config outputs or the configured simulation calibration variable names.

Some dashboards may link simulation records to MP4 files stored on the Perlmutter shared file system, but this behavior is experiment-specific.
