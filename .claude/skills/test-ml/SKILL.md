---
name: test-ml
description: Train ML models (GP, NN, ensemble_NN), populate MLflow, and verify they load and evaluate correctly. Use when testing the ML training and inference pipeline end-to-end.
---

# Test ML Models

Train models with `train_model.py`, then verify they load and evaluate via the `check_model.py` script bundled with this skill (`.claude/skills/test-ml/check_model.py`).

## Parse user intent

- If `$ARGUMENTS` specifies a model type (GP, NN, or ensemble_NN), only test that model type.
- If `$ARGUMENTS` specifies an experiment name, use `experiments/synapse-<name>/config.yaml`.
- If no model type is given, test all three: GP, NN, ensemble_NN.
- If no experiment is given, auto-detect available experiments by globbing `experiments/synapse-*/config.yaml`. Only use experiments whose config contains an `mlflow` section with a `tracking_uri`.

## Pre-flight checks (do these BEFORE any training)

Run all checks and **stop immediately with a clear warning** if any fail:

1. **Conda environment**: Verify `synapse-ml` conda environment exists:
   ```
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate synapse-ml
   ```

2. **Database credentials**: Source `~/db.profile`, then read the config YAML and check that the environment variable named in `database.password_ro_env` is set.

3. **MLflow server**: Read `mlflow.tracking_uri` from the config YAML. Test reachability with a quick socket connection (e.g. `python -c "import socket; socket.create_connection(('<host>', <port>), timeout=5)"`). If unreachable, tell the user to start the MLflow server.

4. **MongoDB**: Test connectivity with a quick pymongo ping:
   ```
   python -c "import pymongo; pymongo.MongoClient(host='<host>', port=<port>, serverSelectionTimeoutMS=5000).admin.command('ping')"
   ```
   Note: for the MongoDB check, source `~/db.profile` first, then read the database credentials from the config file (host, port, auth, username_ro, password_ro_env).

## Train models

Run training from the `ml/` directory. Each command:
```
source ~/db.profile && source ~/miniconda3/etc/profile.d/conda.sh && conda activate synapse-ml && cd <repo_root>/ml && python train_model.py --config_file <config_path> --model <model_type>
```

- When testing **multiple model types**, run the training commands **in parallel** (use background bash tasks).
- When testing a **single model type**, run it in the foreground.
- Wait for all training to complete before proceeding.

## Validate models

After training, run the bundled `check_model.py` for each model type:
```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate synapse-ml && python <repo_root>/.claude/skills/test-ml/check_model.py --config_file <config_path> --model <model_type>
```

## Report results

Provide a summary table:
- Model type | Train status | Validation status
- Show `[PASS]` or `[FAIL]` for each
- For failures, include the error message
