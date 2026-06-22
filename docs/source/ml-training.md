# ML Training

Model training is implemented in `ml/train_model.py`.
It reads config and MongoDB records, trains a model, wraps it with `lume-model`, and optionally registers it in MLflow.

## Model Types

Use `--model` with one of:

- `GP`: Gaussian Process.
- `NN`: single neural network.
- `ensemble_NN`: ensemble neural network. The current ensemble size is defined in `train_nn_ensemble()` in `ml/train_model.py`.

## Command

```bash
python train_model.py --config_file ../experiments/synapse-bella-ip2/config.yaml --model NN
```

Use `--test` to skip MLflow registration.

## Phases

1. Load config, variables, database records, and MLflow settings.
2. Build calibration and normalization transforms.
3. Train on simulation data.
4. Train [calibration](experiment-configuration.md#calibration) on experimental data when available.
5. Build a `lume-model`.
6. Register to MLflow unless `--test` is set.

## MLflow Names

Registered models use:

```text
synapse-<experiment>_<model_type>
```

The MLflow experiment is:

```text
synapse-<experiment>
```

## Pipeline Test

Run against a local MLflow server:

```bash
docker run -p 127.0.0.1:5000:5000 ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0
python tests/test_ml_pipeline.py
```
