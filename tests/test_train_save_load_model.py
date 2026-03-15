#!/usr/bin/env python
"""
Automated test: train ML models, save to MLflow, load and evaluate.

Can be run standalone (CLI) or discovered by pytest.

Usage (standalone):
    python tests/test_train_save_load_model.py
    python tests/test_train_save_load_model.py --model NN --config_file experiments/synapse-bella-ip2
    python tests/test_train_save_load_model.py --test-mlflow-uri http://localhost:5000

Usage (pytest):
    pytest tests/test_train_save_load_model.py -v
    pytest tests/ --test-mlflow-uri http://localhost:5000
"""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from urllib.parse import urlparse

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_TYPES = ["GP", "NN", "ensemble_NN"]
GP_SKIP_THRESHOLD = 1000
ACCURACY_TOLERANCE = 0.20
DEFAULT_MLFLOW_URI = "http://localhost:5000"
CONDA_INIT = "source ~/miniconda3/etc/profile.d/conda.sh"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ML_DIR = _REPO_ROOT / "ml"
_EXPERIMENTS_DIR = _REPO_ROOT / "experiments"
_TESTS_DIR = _REPO_ROOT / "tests"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_mlflow_reachable(uri, timeout=5):
    """Socket-connect to the MLflow server; raise with a clear message if unreachable."""
    parsed = urlparse(uri)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as e:
        raise RuntimeError(
            f"MLflow server at {uri} is not reachable: {e}\n"
            "Start the MLflow server (e.g. `mlflow server --port 5000`) and retry."
        ) from e


def check_db_reachable(cfg):
    """
    Check that the DB password env var is set and that MongoDB responds to a ping.
    Raises RuntimeError with a clear message on failure.
    """
    import pymongo

    db_cfg = cfg.get("database", {})
    password_ro_env = db_cfg.get("password_ro_env")
    if not password_ro_env:
        raise RuntimeError("No 'database.password_ro_env' found in config.")

    password = os.getenv(password_ro_env)
    if password is None:
        raise RuntimeError(
            f"Environment variable '{password_ro_env}' is not set.\n"
            f"Export it before running: export {password_ro_env}=<password>"
        )

    host = db_cfg.get("host", "localhost")
    port = db_cfg.get("port", 27017)
    auth = db_cfg.get("auth", "admin")
    username = db_cfg.get("username_ro")

    try:
        client = pymongo.MongoClient(
            host=host,
            port=port,
            authSource=auth,
            username=username,
            password=password,
            serverSelectionTimeoutMS=5000,
        )
        client.admin.command("ping")
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to MongoDB at {host}:{port}: {e}"
        ) from e


def load_config(path):
    """Accept a directory or .yaml file path; return parsed config dict."""
    p = Path(path)
    if p.is_dir():
        p = p / "config.yaml"
    if not p.exists():
        raise RuntimeError(f"Configuration file not found: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


def get_all_configs():
    """Return list of config.yaml paths for experiments that have an mlflow section."""
    configs = []
    for config_path in sorted(_EXPERIMENTS_DIR.glob("*/config.yaml")):
        try:
            cfg = load_config(config_path)
            if cfg.get("mlflow", {}).get("tracking_uri"):
                configs.append(config_path)
        except Exception:
            pass
    return configs


def count_sim_datapoints(cfg):
    """Count simulation datapoints (experiment_flag=0) in MongoDB. Returns None on error."""
    try:
        import pymongo

        db_cfg = cfg["database"]
        password = os.getenv(db_cfg["password_ro_env"])
        if not password:
            return None
        client = pymongo.MongoClient(
            host=db_cfg["host"],
            port=db_cfg.get("port", 27017),
            authSource=db_cfg.get("auth", "admin"),
            username=db_cfg.get("username_ro"),
            password=password,
            serverSelectionTimeoutMS=5000,
        )
        db = client[db_cfg["name"]]
        date_filter = cfg.get("date_filter", {})
        return db[cfg["experiment"]].count_documents(
            {"experiment_flag": 0, **date_filter}
        )
    except Exception:
        return None


def make_temp_config(cfg, mlflow_uri):
    """
    Write a temporary YAML config with tracking_uri overridden and api_key_env removed.
    Returns the path to the temp file (caller is responsible for cleanup).
    """
    import copy

    tmp_cfg = copy.deepcopy(cfg)
    if "mlflow" not in tmp_cfg:
        tmp_cfg["mlflow"] = {}
    tmp_cfg["mlflow"]["tracking_uri"] = mlflow_uri
    tmp_cfg["mlflow"].pop("api_key_env", None)

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="synapse_test_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(tmp_cfg, f)
    except Exception:
        os.unlink(tmp_path)
        raise
    return tmp_path


def run_in_conda(conda_env, cmd, cwd=None):
    """
    Run `cmd` inside the given conda environment via a login shell.
    Raises subprocess.CalledProcessError on non-zero exit.
    """
    full_cmd = f"{CONDA_INIT} && conda activate {conda_env} && {cmd}"
    result = subprocess.run(
        full_cmd,
        shell=True,
        executable="/bin/bash",
        cwd=str(cwd) if cwd else None,
        capture_output=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, full_cmd)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def preflight(config_dict, mlflow_uri):
    """Run pre-flight checks: MLflow reachability and DB connectivity."""
    check_mlflow_reachable(mlflow_uri)
    check_db_reachable(config_dict)


def run_one_test(config_file, model_type, mlflow_uri=DEFAULT_MLFLOW_URI):
    """
    Full train → save → load → evaluate cycle for one (config, model_type) pair.
    Raises on failure; returns normally on success.
    """
    config_file = Path(config_file)
    if config_file.is_dir():
        config_file = config_file / "config.yaml"

    config_dict = load_config(config_file)

    preflight(config_dict, mlflow_uri)

    # GP: skip if dataset is too large
    if model_type == "GP":
        count = count_sim_datapoints(config_dict)
        if count is not None and count > GP_SKIP_THRESHOLD:
            msg = (
                f"Skipping GP test for '{config_dict.get('experiment')}': "
                f"{count} simulation datapoints > threshold {GP_SKIP_THRESHOLD}."
            )
            warnings.warn(msg)
            # pytest.skip is only available inside pytest; detect it gracefully
            try:
                import pytest
                pytest.skip(msg)
            except ImportError:
                print(f"[SKIP] {msg}")
                return

    tmp_path = make_temp_config(config_dict, mlflow_uri)
    try:
        # Train
        run_in_conda(
            "synapse-ml",
            f"python train_model.py --config_file {tmp_path} --model {model_type}",
            cwd=_ML_DIR,
        )

        # Load and evaluate
        run_in_conda(
            "synapse-ml",
            f"python {_TESTS_DIR / 'check_model.py'} --config_file {tmp_path} --model {model_type}",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Pytest interface
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--test-mlflow-uri",
        default=DEFAULT_MLFLOW_URI,
        help=f"MLflow tracking URI to use during tests (default: {DEFAULT_MLFLOW_URI})",
    )


def pytest_generate_tests(metafunc):
    if "config_file" in metafunc.fixturenames and "model_type" in metafunc.fixturenames:
        configs = get_all_configs()
        params = [
            (str(cfg), model)
            for cfg in configs
            for model in MODEL_TYPES
        ]
        ids = [
            f"{Path(cfg).parent.name}-{model}"
            for cfg, model in params
        ]
        metafunc.parametrize("config_file,model_type", params, ids=ids)


try:
    import pytest

    @pytest.fixture(scope="session")
    def mlflow_uri(request):
        return request.config.getoption("--test-mlflow-uri", default=DEFAULT_MLFLOW_URI)

    def test_train_save_load(config_file, model_type, mlflow_uri):
        run_one_test(config_file, model_type, mlflow_uri)

except ImportError:
    pass  # pytest not available; CLI mode only


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train, save, and load ML models end-to-end."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_TYPES,
        default=None,
        help="Model type to test. Defaults to all.",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        help="Path to a config.yaml file or experiment directory. Defaults to all.",
    )
    parser.add_argument(
        "--test-mlflow-uri",
        default=DEFAULT_MLFLOW_URI,
        dest="mlflow_uri",
        help=f"MLflow tracking URI (default: {DEFAULT_MLFLOW_URI})",
    )
    args = parser.parse_args()

    # Resolve model types
    models_to_test = [args.model] if args.model else MODEL_TYPES

    # Resolve config files
    if args.config_file:
        configs_to_test = [Path(args.config_file)]
        if configs_to_test[0].is_dir():
            configs_to_test = [configs_to_test[0] / "config.yaml"]
    else:
        configs_to_test = get_all_configs()

    if not configs_to_test:
        print("No config files found with an mlflow.tracking_uri section.")
        sys.exit(1)

    results = []
    for config_path in configs_to_test:
        exp_name = config_path.parent.name
        for model_type in models_to_test:
            print(f"\n{'='*60}")
            print(f"Testing: {exp_name} / {model_type}")
            print(f"{'='*60}")
            try:
                run_one_test(config_path, model_type, args.mlflow_uri)
                results.append((exp_name, model_type, "PASS", ""))
            except SystemExit:
                # pytest.skip or explicit skip
                results.append((exp_name, model_type, "SKIP", ""))
            except Exception as e:
                results.append((exp_name, model_type, "FAIL", str(e)))
                print(f"[FAIL] {e}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    col_w = max(len(r[0]) for r in results) + 2
    header = f"{'Experiment':<{col_w}} {'Model':<14} {'Status'}"
    print(header)
    print("-" * len(header))
    for exp_name, model_type, status, error in results:
        line = f"{exp_name:<{col_w}} {model_type:<14} [{status}]"
        if error:
            line += f"  {error[:80]}"
        print(line)

    any_fail = any(r[2] == "FAIL" for r in results)
    sys.exit(1 if any_fail else 0)
