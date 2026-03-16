#!/usr/bin/env python
"""
Check that a model stored in MLflow loads and evaluates correctly,
using the same logic as the dashboard.

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


def load_config(config_file):
    if not os.path.exists(config_file):
        raise RuntimeError(f"Configuration file not found: {config_file}")
    with open(config_file) as f:
        return yaml.safe_load(f.read())


def load_experimental_data(config_dict):
    """Fetch all experimental points from the database."""
    input_names = [v["name"] for v in config_dict["inputs"].values()]
    output_names = [v["name"] for v in config_dict["outputs"].values()]

    db = load_database(config_dict)
    exp_data, _ = load_data(db, config_dict["experiment"])

    return exp_data, input_names, output_names


def check_evaluate(config_dict, model_type):
    """Load model and evaluate with experimental data; verify accuracy (relative RMSE <= 25%)."""
    # Load model
    mm = ModelManager(config_dict=config_dict, model_type_tag=model_type)
    if not mm.avail():
        raise RuntimeError(f"Model '{model_type}' could not be loaded from MLflow.")
    # Load experimental data
    df_exp, input_names, output_names = load_experimental_data(config_dict)
    # Convert input to the format expected by the model manager
    inputs = {n: torch.tensor(df_exp[n].values) for n in input_names}

    # Check accuracy
    all_passed = True
    for output_name in output_names:
        prediction, _, _ = mm.evaluate(inputs, output_name)
        actual = torch.tensor(df_exp[output_name].values)
        rel_errors = (prediction - actual) / torch.max(
            torch.abs(actual), torch.abs(prediction)
        )
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
    parser = argparse.ArgumentParser(
        description="Verify that an MLflow model loads and evaluates correctly."
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

    # Load configuration
    config_dict = load_config(args.config_file)
    print(f"Experiment: {config_dict['experiment']}")

    # Load model and evaluate with experimental data
    try:
        check_evaluate(config_dict, args.model)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("[PASS] Model loaded and evaluated successfully.")
