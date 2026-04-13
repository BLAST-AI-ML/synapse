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
import torch
import yaml

DASHBOARD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard"
)  # similar to "cd ../dashboard"
sys.path.insert(0, DASHBOARD_DIR)
from calibration_manager import SimulationCalibrationManager  # noqa: E402
from model_manager import ModelManager  # noqa: E402
from state_manager import state  # noqa: E402
from utils import load_database, load_data as _load_data  # noqa: E402


MODEL_TYPES = ["GP", "NN", "ensemble_NN"]
ACCURACY_TOLERANCE = 0.80


def load_data(config_dict):
    """Fetch all experimental and simulation points from the database."""
    input_names = [v["name"] for v in config_dict["inputs"].values()]
    output_names = [v["name"] for v in config_dict["outputs"].values()]

    db = load_database(config_dict)
    exp_data, sim_data = _load_data(db, config_dict["experiment"])

    return exp_data, sim_data, input_names, output_names


def check_accuracy(mm, df, input_names, output_names, label):
    """Evaluate model on *df* and return True if all outputs pass the accuracy threshold."""
    if len(df) == 0:
        print(f"[SKIP] No {label} data available; skipping accuracy check.")
        return True

    inputs = {n: torch.tensor(df[n].values) for n in input_names}

    all_passed = True
    for output_name in output_names:
        actual = torch.tensor(df[output_name].values)
        if actual.isnan().all():
            print(
                f"  [SKIP] Output '{output_name}': all actual values are NaN; skipping."
            )
            continue
        prediction, _, _ = mm.evaluate(inputs, output_name)
        rel_errors = (prediction - actual) / torch.max(
            torch.abs(actual), torch.abs(prediction)
        )
        rmse = torch.sqrt(torch.nanmean(rel_errors**2)).item()
        if rmse <= ACCURACY_TOLERANCE:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False
        print(
            f"  [{status}] Output '{output_name}': relative RMSE = {rmse:.1%} (tolerance {ACCURACY_TOLERANCE:.0%})"
        )
    return all_passed


def check_evaluate(config_dict, model_type):
    """Load model and evaluate with experimental and simulation data; verify accuracy."""
    # Set up calibration so ModelManager can populate inferred values
    simulation_calibration = config_dict.get("simulation_calibration", {})
    cal_manager = SimulationCalibrationManager(simulation_calibration)
    state.use_inferred_calibration = True

    # Load model (populates inferred calibration in state.simulation_calibration)
    mm = ModelManager(config_dict=config_dict, model_type=model_type)

    # Load experimental and simulation data
    df_exp, df_sim, input_names, output_names = load_data(config_dict)

    # Check accuracy on experimental data
    print("Checking experimental data...")
    exp_passed = check_accuracy(mm, df_exp, input_names, output_names, "experimental")

    # Convert simulation data to experimental units using inferred calibration
    cal_manager.convert_sim_to_exp(df_sim)

    # Check accuracy on simulation data
    print("Checking simulation data...")
    sim_passed = check_accuracy(mm, df_sim, input_names, output_names, "simulation")

    if not (exp_passed and sim_passed):
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
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    print(f"Experiment: {config_dict['experiment']}")

    # Load model and evaluate with experimental and simulation data
    try:
        check_evaluate(config_dict, args.model)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("[PASS] Model loaded and evaluated successfully.")
