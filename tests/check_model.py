#!/usr/bin/env python
"""
Check that a model stored in MLflow loads and evaluates correctly,
using the same download logic as the dashboard.

Usage:
    python check_model.py --config_file <path/to/config.yaml> --model <GP|NN|ensemble_NN>
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import pymongo
import torch
import yaml
import mlflow

_DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(_DASHBOARD_DIR))
from model_manager import enable_amsc_x_api_key


MODEL_TYPES = ["GP", "NN", "ensemble_NN"]
ACCURACY_TOLERANCE = 0.25


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


def download_model(config_dict, model_type):
    """Download the model from MLflow, exactly as the dashboard does."""
    if "mlflow" not in config_dict or not config_dict["mlflow"].get("tracking_uri"):
        raise RuntimeError(
            "No mlflow.tracking_uri found in config file; cannot load model from MLflow."
        )

    tracking_uri = config_dict["mlflow"]["tracking_uri"]
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


def load_experimental_data(config_dict):
    """Fetch all experimental points from the database; return input dict and output DataFrame."""
    experiment = config_dict["experiment"]
    input_variables = config_dict["inputs"]
    input_names = [v["name"] for v in input_variables.values()]
    output_variables = config_dict["outputs"]
    output_names = [v["name"] for v in output_variables.values()]

    db_cfg = config_dict["database"]
    db_password = os.getenv(db_cfg["password_ro_env"])
    if db_password is None:
        raise RuntimeError(f"Environment variable {db_cfg['password_ro_env']} must be set!")
    db = pymongo.MongoClient(
        host=db_cfg["host"],
        port=db_cfg["port"],
        authSource=db_cfg["auth"],
        username=db_cfg["username_ro"],
        password=db_password,
        directConnection=(db_cfg["host"] in ["localhost", "127.0.0.1"]),
    )[db_cfg["name"]]

    date_filter = config_dict.get("date_filter", {})
    df_exp = pd.DataFrame(db[experiment].find({"experiment_flag": 1, **date_filter}))

    if df_exp.empty:
        return None, None, output_names

    missing = [name for name in input_names + output_names if name not in df_exp.columns]
    if missing:
        raise RuntimeError(f"Missing columns in experimental data: {missing}")

    print(f"Fetched {len(df_exp)} experimental points from the database.")

    inputs = {
        name: torch.tensor(df_exp[name].values, dtype=torch.float64)
        for name in input_names
    }
    return inputs, df_exp, output_names


def check_evaluate(model, config_dict):
    """Call evaluate() with experimental data and verify accuracy (relative RMSE <= 20%)."""
    inputs, df_exp, output_names = load_experimental_data(config_dict)

    if inputs is None:
        print("[WARN] No experimental points found in the database; skipping accuracy check.")
        return None

    n_points = len(next(iter(inputs.values())))
    print(f"Calling model.evaluate() with {n_points} experimental points...")
    print(f"Inputs: {inputs}")
    result = model.evaluate(inputs)
    print("evaluate() succeeded.")
    print(f"Output keys: {list(result.keys())}")

    # Accuracy check: RMSE <= ACCURACY_TOLERANCE for each output
    all_passed = True
    for output_name in output_names:
        if output_name not in result:
            print(f"[WARN] Output '{output_name}' not found in evaluate() result; skipping.")
            continue

        raw = result[output_name]
        # GP / ensemble_NN return a distribution; NN returns a tensor directly
        if torch.is_tensor(raw):
            predicted = raw.float()
        elif hasattr(raw, "mean"):
            predicted = raw.mean.detach().float()
        else:
            predicted = torch.tensor(raw, dtype=torch.float)

        print(f"Predicted: {predicted}")
        print(f"Predicted mean: {predicted.mean()}")
        actual = torch.tensor(df_exp[output_name].values, dtype=torch.float)
        print(f"Actual: {actual}")
        print(f"Actual mean: {actual.mean()}")



        rel_errors = (predicted - actual) / torch.max(torch.abs(actual), torch.abs(predicted))
        print(f"Rel errors: {rel_errors}")
        rmse = torch.sqrt((rel_errors ** 2).mean()).item()
        status = "PASS" if rmse <= ACCURACY_TOLERANCE else "FAIL"
        print(f"  [{status}] Output '{output_name}': relative RMSE = {rmse:.1%} (threshold {ACCURACY_TOLERANCE:.0%})")
        if rmse > ACCURACY_TOLERANCE:
            all_passed = False

    if not all_passed:
        raise RuntimeError(
            f"Accuracy check failed: relative RMSE exceeded {ACCURACY_TOLERANCE:.0%} for one or more outputs."
        )

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

    # Evaluate with experimental data and check accuracy
    try:
        check_evaluate(model, config_dict)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("[PASS] Model loaded and evaluated successfully.")
