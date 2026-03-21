#!/usr/bin/env python
"""
Automated test: train ML models, save to MLflow, load and evaluate.

Exercises the full ML lifecycle: training -> upload to MLflow -> download -> accuracy check.

Requires a local, empty MLflow server. Start one before running, e.g. with Docker:
    docker run -p 127.0.0.1:5000:5000 ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0

This script makes a temporary copy of each config file with the MLflow URI replaced by the
test server URI, so it never touches the production MLflow server.

Can be run standalone (CLI).

Usage (standalone):
    python tests/test_train_save_load_model.py
    python tests/test_train_save_load_model.py --model NN --config_file experiments/synapse-bella-ip2
    python tests/test_train_save_load_model.py --test-mlflow-uri http://localhost:5000
"""

import argparse
import copy
import glob
import os
import socket
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

import yaml

# Constants

MODEL_TYPES = ["GP", "NN", "ensemble_NN"]
DEFAULT_MLFLOW_URI = "http://localhost:5000"
CONDA_INIT = "source ~/miniconda3/etc/profile.d/conda.sh"  # Needed in order to use conda in the subprocesses

# GP training takes too long above this number of simulation datapoints
GP_SKIP_THRESHOLD = 1000

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # similar to "cd ../.."
ML_DIR = os.path.join(REPO_ROOT, "ml")
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

# Helpers


def check_mlflow_reachable(uri, timeout=5):
    """Socket-connect to the MLflow server; raise with a clear message if unreachable."""
    parsed = urlparse(uri)
    host = parsed.hostname
    port = parsed.port
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as e:
        raise RuntimeError(
            f"MLflow server at {uri} is not reachable: {e}\n"
            "Start a local MLflow server and retry, e.g.:\n"
            "  docker run -p 127.0.0.1:5000:5000 ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0"
        ) from e


def check_db_reachable(cfg):
    """Check that the DB password env var is set and that MongoDB responds to a ping."""
    import pymongo

    db_cfg = cfg["database"]
    password_ro_env = db_cfg["password_ro_env"]
    password = os.getenv(password_ro_env)
    if password is None:
        raise RuntimeError(
            f"Environment variable '{password_ro_env}' is not set.\n"
            f"Export it before running: export {password_ro_env}=<password>"
        )

    try:
        client = pymongo.MongoClient(
            host=db_cfg["host"],
            port=db_cfg["port"],
            authSource=db_cfg["auth"],
            username=db_cfg["username_ro"],
            password=password,
            serverSelectionTimeoutMS=5000,  # fail after 5 seconds if DB is unreachable
        )
        client.admin.command("ping")
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to MongoDB at {db_cfg['host']}:{db_cfg['port']}: {e}"
        ) from e



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
            port=db_cfg["port"],
            authSource=db_cfg["auth"],
            username=db_cfg["username_ro"],
            password=password,
            serverSelectionTimeoutMS=5000,  # fail after 5 seconds if DB is unreachable
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
    tmp_cfg = copy.deepcopy(cfg)
    tmp_cfg["mlflow"]["tracking_uri"] = mlflow_uri
    tmp_cfg["mlflow"].pop("api_key_env", None)

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="synapse_test_")
    with os.fdopen(fd, "w") as f:
        yaml.dump(tmp_cfg, f)
    return tmp_path


def run_in_conda(conda_env, cmd, cwd=None):
    """Run `cmd` inside the given conda environment via a login shell."""
    full_cmd = f"{CONDA_INIT} && conda activate {conda_env} && {cmd}"
    result = subprocess.run(
        full_cmd,
        shell=True,
        executable="/bin/bash",
        cwd=cwd,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, full_cmd)


def run_one_test(config_file, model_type, mlflow_uri=DEFAULT_MLFLOW_URI):
    """
    Full train, save, load, and evaluate cycle for one (config, model_type) pair.
    Raises on failure; returns normally on success.
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    check_mlflow_reachable(mlflow_uri)
    check_db_reachable(cfg)

    # GP training is too slow on large datasets; skip if above threshold
    if model_type == "GP":
        count = count_sim_datapoints(cfg)
        if count is not None and count > GP_SKIP_THRESHOLD:
            print(
                f"[SKIP] GP test for '{cfg['experiment']}': "
                f"{count} simulation datapoints > threshold {GP_SKIP_THRESHOLD}."
            )
            return

    tmp_path = make_temp_config(cfg, mlflow_uri)
    try:
        run_in_conda(
            "synapse-ml",
            f"python train_model.py --config_file {tmp_path} --model {model_type}",
            cwd=ML_DIR,
        )
        run_in_conda(
            "synapse-gui",
            f"python {os.path.join(TESTS_DIR, 'check_model.py')} --config_file {tmp_path} --model {model_type}",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train, save, and load ML models end-to-end against a local test MLflow server.\n"
            "Requires a local MLflow server running before execution, e.g.:\n"
            "  docker run -p 127.0.0.1:5000:5000 ghcr.io/mlflow/mlflow mlflow server --host 0.0.0.0\n"
            "Config files are copied temporarily with the MLflow URI replaced by the test server URI, "
            "so the production MLflow server is never touched."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    models_to_test = [args.model] if args.model else MODEL_TYPES

    if args.config_file:
        p = args.config_file
        configs_to_test = [os.path.join(p, "config.yaml") if os.path.isdir(p) else p]
    else:
        configs_to_test = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "*/config.yaml")))

    if not configs_to_test:
        print("No config files found with an mlflow.tracking_uri section.")
        sys.exit(1)

    results = []
    for config_path in configs_to_test:
        exp_name = os.path.basename(os.path.dirname(config_path))
        for model_type in models_to_test:
            print(f"\n{'=' * 60}")
            print(f"Testing: {exp_name} / {model_type}")
            print(f"{'=' * 60}")
            try:
                run_one_test(config_path, model_type, args.mlflow_uri)
                results.append((exp_name, model_type, "PASS", ""))
            except SystemExit:
                results.append((exp_name, model_type, "SKIP", ""))
            except Exception as e:
                results.append((exp_name, model_type, "FAIL", str(e)))
                print(f"[FAIL] {e}")

    print("\nSUMMARY")
    for exp_name, model_type, status, error in results:
        print(
            f"  [{status}] {exp_name} / {model_type}" + (f": {error}" if error else "")
        )

    any_fail = any(r[2] == "FAIL" for r in results)
    sys.exit(1 if any_fail else 0)
