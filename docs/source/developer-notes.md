# Developer Notes

## Style

Python code is linted and formatted with Ruff through pre-commit:

```bash
pre-commit run --files <modified files>
```

Ruff runs with its default rule set; there is no `pyproject.toml` or `ruff.toml` to override it.

## Environments

- Dashboard dependencies live in `dashboard/environment.yml`.
- ML dependencies live in `ml/environment.yml`.
- Regenerate the matching `environment-lock.yml` after dependency changes.

## Testing

The project does not have a full pytest suite.
The main integration check is:

```bash
python tests/test_ml_pipeline.py
```

It requires a local MLflow server.

## Patterns

- Dashboard features use manager classes in `dashboard/*_manager.py`.
- Experiment-specific behavior belongs under `experiments/synapse-*`.
- Shared dashboard helpers live in `dashboard/utils.py`.
