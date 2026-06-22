# Experiment Configuration

An experiment is a directory named `experiments/synapse-<experiment>/`.
The dashboard strips `synapse-` and uses the rest as the experiment identifier.

Each experiment should provide:

- `config.yaml`
- optional `simulation_scripts/`
- optional `experiment_scripts/`

## Required Config Sections

- `experiment`: collection and model namespace, for example `bella-ip2`.
- `database`: MongoDB connection and credential environment variables.
- `mlflow`: tracking URI and optional API key environment variable.
- `execution_mode`: ML training and simulation mode hints.
- `inputs`: scalar variables with `name`, `type`, `default`, and `value_range`.
- `outputs`: scalar variables with `name` and `type`.

## Calibration

`simulation_calibration` maps simulation variable names to experimental variable names:

```yaml
simulation_calibration:
  input1:
    name: "simulation_variable"
    unit: "unit"
    depends_on: "experimental_variable"
    alpha_guess: 1.0
    alpha_uncertainty: 0.1
    beta_guess: 0.0
    beta_uncertainty: 0.0
```

Dashboard display uses:

```text
experimental = simulation / alpha + beta
```

Simulation launch uses:

```text
simulation = alpha * (experimental - beta)
```

These are inverse conversions: display maps simulation to experimental units, while launch maps dashboard parameters back to simulation units.

## Add an Experiment

1. Clone or create `experiments/synapse-<experiment>/`.
2. Add `config.yaml`.
3. Ensure MongoDB fields match the configured input and output variable names.
4. Add `simulation_scripts/` only if dashboard launch is needed.
5. Train and register a model if dashboard predictions are needed.
