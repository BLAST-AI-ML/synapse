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
import torch
import yaml

_DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(_DASHBOARD_DIR))
from model_manager import ModelManager
from utils import load_database, load_data


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


def load_experimental_data(config_dict):
    """Fetch all experimental points from the database; return input dict and output DataFrame."""
    input_names = [v["name"] for v in config_dict["inputs"].values()]
    output_names = [v["name"] for v in config_dict["outputs"].values()]

    db = load_database(config_dict)
    exp_data, _ = load_data(db, config_dict["experiment"])

    if exp_data.empty:
        return None, None, output_names

    missing = [n for n in input_names + output_names if n not in exp_data.columns]
    if missing:
        raise RuntimeError(f"Missing columns in experimental data: {missing}")

    print(f"Fetched {len(exp_data)} experimental points from the database.")
    inputs = {
        n: torch.tensor(exp_data[n].values, dtype=torch.float64) for n in input_names
    }
    return inputs, exp_data, output_names


def check_evaluate(config_dict, model_type):
    """Load model and evaluate with experimental data; verify accuracy (relative RMSE <= 25%)."""
    mm = ModelManager(config_dict=config_dict, model_type_tag=model_type)
    if not mm.avail():
        raise RuntimeError(f"Model '{model_type}' could not be loaded from MLflow.")

    inputs, df_exp, output_names = load_experimental_data(config_dict)

    if inputs is None:
        print(
            "[WARN] No experimental points found in the database; skipping accuracy check."
        )
        return

    n_points = len(next(iter(inputs.values())))
    print(f"Calling mm.evaluate() with {n_points} experimental points...")

    all_passed = True
    for output_name in output_names:
        mean, _, _ = mm.evaluate(inputs, output_name)
        if not torch.is_tensor(mean):
            mean = torch.tensor(mean, dtype=torch.float)
        else:
            mean = mean.float()

        actual = torch.tensor(df_exp[output_name].values, dtype=torch.float)
        rel_errors = (mean - actual) / torch.max(torch.abs(actual), torch.abs(mean))
        rmse = torch.sqrt((rel_errors**2).mean()).item()
        status = "PASS" if rmse <= ACCURACY_TOLERANCE else "FAIL"
        print(
            f"  [{status}] Output '{output_name}': relative RMSE = {rmse:.1%} (threshold {ACCURACY_TOLERANCE:.0%})"
        )
        if rmse > ACCURACY_TOLERANCE:
            all_passed = False

    if not all_passed:
        raise RuntimeError(
            f"Accuracy check failed: relative RMSE exceeded {ACCURACY_TOLERANCE:.0%} for one or more outputs."
        )


if __name__ == "__main__":
    config_file, model_type = parse_arguments()

    # Load configuration
    config_dict = load_config(config_file)
    print(f"Experiment: {config_dict['experiment']}")

    # Load model and evaluate with experimental data
    try:
        check_evaluate(config_dict, model_type)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("[PASS] Model loaded and evaluated successfully.")
