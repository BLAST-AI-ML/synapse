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
    python tests/test_my_pipeline.py
    python tests/test_my_pipeline.py --model NN --config_file experiments/synapse-bella-ip2/config.yaml
    python tests/test_my_pipeline.py --test-mlflow-uri http://localhost:5000
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
    """Count simulation datapoints (experiment_flag=0) in MongoDB."""
    import pymongo

    db_cfg = cfg["database"]
    password = os.getenv(db_cfg["password_ro_env"])
    client = pymongo.MongoClient(
        host=db_cfg["host"],
        port=db_cfg["port"],
        authSource=db_cfg["auth"],
        username=db_cfg["username_ro"],
        password=password,
    )
    db = client[db_cfg["name"]]
    date_filter = cfg.get("date_filter", {})
    return db[cfg["experiment"]].count_documents({"experiment_flag": 0, **date_filter})


def override_mlflow_config(cfg, mlflow_uri):
    """Return a copy of cfg with tracking_uri overridden and api_key_env removed."""
    tmp_cfg = copy.deepcopy(cfg)
    tmp_cfg["mlflow"]["tracking_uri"] = mlflow_uri
    tmp_cfg["mlflow"].pop("api_key_env", None)
    return tmp_cfg


def run_one_test(config_file, model_type, mlflow_uri=DEFAULT_MLFLOW_URI) -> str:
    """
    Full train, save, load, and evaluate cycle for one (config, model_type) pair.
    Returns "PASS" or "SKIP". Raises on failure.
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    check_mlflow_reachable(mlflow_uri)
    check_db_reachable(cfg)

    # GP training is too slow on large datasets; skip if above threshold
    if model_type == "GP":
        count = count_sim_datapoints(cfg)
        if count > GP_SKIP_THRESHOLD:
            reason = f"{count} simulation datapoints > threshold {GP_SKIP_THRESHOLD}"
            print(f"[SKIP] GP test for '{cfg['experiment']}': {reason}.")
            return "SKIP"

    tmp_cfg = override_mlflow_config(cfg, mlflow_uri)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "config.yaml")
        with open(tmp_path, "w") as f:
            yaml.dump(tmp_cfg, f)
        subprocess.run(
            f"conda run -n synapse-ml python train_model.py --config_file {tmp_path} --model {model_type}",
            shell=True,
            check=True,
            cwd=ML_DIR,
        )
        subprocess.run(
            f"conda run -n synapse-gui python check_model.py --config_file {tmp_path} --model {model_type}",
            shell=True,
            check=True,
            cwd=TESTS_DIR,
        )
    return "PASS"


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
        help="Path to a config.yaml file. Defaults to all available config.yaml files in the experiments directory.",
    )
    parser.add_argument(
        "--test-mlflow-uri",
        default=DEFAULT_MLFLOW_URI,
        dest="mlflow_uri",
        help=f"MLflow tracking URI (default: {DEFAULT_MLFLOW_URI})",
    )
    args = parser.parse_args()

    # Check which models to test
    models_to_test = [args.model] if args.model else MODEL_TYPES

    # Check which config files to test
    if args.config_file:
        p = args.config_file
        configs_to_test = [p]
    else:
        configs_to_test = sorted(
            glob.glob(os.path.join(EXPERIMENTS_DIR, "*/config.yaml"))
        )
    if not configs_to_test:
        print("No config files found.")
        sys.exit(1)

    results = []
    for config_path in configs_to_test:
        exp_name = os.path.basename(os.path.dirname(config_path))
        for model_type in models_to_test:
            print(f"\n{'=' * 60}")
            print(f"Testing: {exp_name} / {model_type}")
            print(f"{'=' * 60}")
            try:
                status = run_one_test(config_path, model_type, args.mlflow_uri)
                results.append((exp_name, model_type, status, ""))
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
