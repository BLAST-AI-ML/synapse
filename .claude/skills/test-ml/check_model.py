#!/usr/bin/env python
"""
Check that a model stored in MLflow loads and evaluates correctly,
using the same download logic as the dashboard.

Usage:
    python check_model.py --config_file <path/to/config.yaml> --model <GP|NN|ensemble_NN>
"""

import argparse
import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import torch
import yaml
import mlflow

# Import DB connection helper from train_model.py
_ML_DIR = Path(__file__).resolve().parents[3] / "ml"
sys.path.insert(0, str(_ML_DIR))
from train_model import connect_to_db


MODEL_TYPES = ["GP", "NN", "ensemble_NN"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Verify that an MLflow model loads and predicts with default parameters."
    )
    parser.add_argument(
        "--config_file",
        help="Path to the configuration file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Model type: GP, NN, or ensemble_NN",
        choices=MODEL_TYPES,
        required=True,
    )
    args = parser.parse_args()
    print(f"Config file: {args.config_file}, Model type: {args.model}")
    return args.config_file, args.model


def load_config(config_file):
    if not os.path.exists(config_file):
        raise RuntimeError(f"Configuration file not found: {config_file}")
    with open(config_file) as f:
        return yaml.safe_load(f.read())


def enable_amsc_x_api_key(config_dict):
    """Inject AmSC X-Api-Key header into all MLflow requests (mirrors train_model.py)."""
    import mlflow.utils.rest_utils as rest_utils

    mlflow_cfg = config_dict.get("mlflow") or {}
    api_key_env = mlflow_cfg.get("api_key_env")
    if not api_key_env:
        raise KeyError(
            "Missing 'api_key_env' in 'mlflow' configuration for AmSC authentication."
        )
    api_key = os.getenv(api_key_env)
    if api_key is None:
        raise KeyError(
            f"Environment variable '{api_key_env}' (from mlflow.api_key_env) is not set."
        )

    _orig = rest_utils.http_request

    def patched(host_creds, endpoint, method, *args, **kwargs):
        if "headers" in kwargs and kwargs["headers"] is not None:
            h = dict(kwargs["headers"])
            h["X-Api-Key"] = api_key
            kwargs["headers"] = h
        else:
            h = dict(kwargs.get("extra_headers") or {})
            h["X-Api-Key"] = api_key
            kwargs["extra_headers"] = h
        return _orig(host_creds, endpoint, method, *args, **kwargs)

    rest_utils.http_request = patched


def check_server_reachable(tracking_uri, timeout=5):
    """Quick socket check to fail fast if the MLflow server is unreachable."""
    parsed = urlparse(tracking_uri)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        print(f"MLflow server reachable at {host}:{port}")
    except OSError as e:
        raise RuntimeError(
            f"MLflow server at {tracking_uri} is not reachable: {e}"
        ) from e


def download_model(config_dict, model_type):
    """Download the model from MLflow, exactly as the dashboard does."""
    if "mlflow" not in config_dict or not config_dict["mlflow"].get("tracking_uri"):
        raise RuntimeError(
            "No mlflow.tracking_uri found in config file; cannot load model from MLflow."
        )

    tracking_uri = config_dict["mlflow"]["tracking_uri"]
    check_server_reachable(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    # Mirror dashboard authentication logic
    if tracking_uri == "https://mlflow.american-science-cloud.org":
        enable_amsc_x_api_key(config_dict)

    experiment = config_dict["experiment"]
    model_name = f"{experiment}_{model_type}"
    model_uri = f"models:/{model_name}/latest"
    print(f"Downloading model '{model_uri}' ...")

    # Same download command as in the dashboard (model_manager.py)
    model = (
        mlflow.pyfunc.load_model(model_uri)
        .unwrap_python_model()
        .model
    )
    print(f"Model downloaded successfully: {type(model).__name__}")
    return model


def load_experimental_inputs(config_dict):
    """Fetch all experimental points from the database and return as a batch input dict."""
    experiment = config_dict["experiment"]
    input_variables = config_dict["inputs"]
    input_names = [v["name"] for v in input_variables.values()]

    db = connect_to_db(config_dict)
    date_filter = config_dict.get("date_filter", {})
    df_exp = pd.DataFrame(db[experiment].find({"experiment_flag": 1, **date_filter}))

    if df_exp.empty:
        raise RuntimeError("No experimental points found in the database.")

    missing = [name for name in input_names if name not in df_exp.columns]
    if missing:
        raise RuntimeError(f"Missing input columns in experimental data: {missing}")

    print(f"Fetched {len(df_exp)} experimental points from the database.")
    return {
        name: torch.tensor(df_exp[name].values, dtype=torch.float64)
        for name in input_names
    }


def check_evaluate(model, config_dict):
    """Call evaluate() with experimental data fetched from the database."""
    inputs = load_experimental_inputs(config_dict)
    print(f"Calling model.evaluate() with {len(next(iter(inputs.values())))} experimental points...")
    result = model.evaluate(inputs)
    print("evaluate() succeeded.")
    print(f"Output keys: {list(result.keys())}")
    return result


if __name__ == "__main__":
    config_file, model_type = parse_arguments()

    # Load configuration
    config_dict = load_config(config_file)
    print(f"Experiment: {config_dict['experiment']}")

    # Download model from MLflow
    try:
        model = download_model(config_dict, model_type)
    except Exception as e:
        print(f"[FAIL] Could not download model: {e}")
        sys.exit(1)

    # Evaluate with default parameters
    try:
        check_evaluate(model, config_dict)
    except Exception as e:
        print(f"[FAIL] evaluate() raised an error: {e}")
        sys.exit(1)

    print("[PASS] Model loaded and evaluated successfully.")
