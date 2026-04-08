#!/usr/bin/env python
"""
Check that a model stored in MLflow loads and evaluates correctly,
using the same logic as the dashboard.

Usage:
    python check_model.py --config_file <path/to/config.yaml> --model <GP|NN|ensemble_NN>
"""

import argparse
import copy
import os
import sys
import torch
import yaml

DASHBOARD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard"
)  # similar to "cd ../dashboard"
sys.path.insert(0, DASHBOARD_DIR)
from model_manager import ModelManager  # noqa: E402
from utils import load_database, load_data  # noqa: E402


MODEL_TYPES = ["GP", "NN", "ensemble_NN"]
ACCURACY_TOLERANCE = 0.80


def load_experimental_data(config_dict):
    """Fetch all experimental points from the database."""
    input_names = [v["name"] for v in config_dict["inputs"].values()]
    output_names = [v["name"] for v in config_dict["outputs"].values()]

    db = load_database(config_dict)
    exp_data, _ = load_data(db, config_dict["experiment"])

    return exp_data, input_names, output_names


def load_sim_data(config_dict):
    """Fetch all simulation points from the database."""
    input_names = [v["name"] for v in config_dict["inputs"].values()]
    output_names = [v["name"] for v in config_dict["outputs"].values()]
    sim_cal = config_dict.get("simulation_calibration", {})
    sim_input_names = [
        v["name"] for v in sim_cal.values() if v["depends_on"] in input_names
    ]
    sim_output_names = [
        v["name"] for v in sim_cal.values() if v["depends_on"] in output_names
    ]

    db = load_database(config_dict)
    _, sim_data = load_data(db, config_dict["experiment"])

    return sim_data, sim_input_names, sim_output_names


def check_evaluate_exp(config_dict, model_type):
    """Load model and evaluate with experimental data; verify accuracy (relative RMSE <= threshold)."""
    # Load model
    mm = ModelManager(config_dict=config_dict, model_type=model_type)
    # Load experimental data
    df_exp, input_names, output_names = load_experimental_data(config_dict)

    # Skip accuracy check if no experimental data available
    if len(df_exp) == 0:
        print(
            f"[SKIP] No experimental data available for {config_dict['experiment']}; skipping accuracy check."
        )
        return

    # Convert input to the format expected by the model manager
    inputs = {n: torch.tensor(df_exp[n].values) for n in input_names}

    # Check accuracy
    all_passed = True
    for output_name in output_names:
        actual = torch.tensor(df_exp[output_name].values)
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

    if not all_passed:
        raise RuntimeError(
            f"Accuracy check failed: relative RMSE exceeded {ACCURACY_TOLERANCE:.0%} for one or more outputs."
        )


def check_evaluate_sim(config_dict, model_type):
    """Load model (without exp<->sim calibration transformers) and evaluate on simulation data."""
    if not config_dict.get("simulation_calibration"):
        print(
            "[SKIP] No 'simulation_calibration' in config; skipping simulation accuracy check."
        )
        return

    # Load model
    mm = ModelManager(config_dict=config_dict, model_type=model_type)
    # Load simulation data and sim variable names
    df_sim, sim_input_names, sim_output_names = load_sim_data(config_dict)

    if len(df_sim) == 0:
        print(
            f"[SKIP] No simulation data available for {config_dict['experiment']}; skipping simulation accuracy check."
        )
        return

    # Create a copy of the underlying model without the exp<->sim calibration transformers
    model_sim = copy.deepcopy(mm._ModelManager__model)
    model_sim.input_transformers = model_sim.input_transformers[
        1:
    ]  # skip exp->sim calibration
    model_sim.output_transformers = model_sim.output_transformers[
        :-1
    ]  # skip sim->exp calibration

    # Convert sim inputs to the format expected by the model
    inputs = {n: torch.tensor(df_sim[n].values) for n in sim_input_names}

    # Check accuracy
    all_passed = True
    for sim_out_name in sim_output_names:
        actual = torch.tensor(df_sim[sim_out_name].values)
        if actual.isnan().all():
            print(
                f"  [SKIP] Sim output '{sim_out_name}': all actual values are NaN; skipping."
            )
            continue
        output_dict = model_sim.evaluate(inputs)
        if model_type == "NN":
            prediction = output_dict[sim_out_name]
        else:
            prediction = output_dict[sim_out_name].mean.detach()
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
            f"  [{status}] Sim output '{sim_out_name}': relative RMSE = {rmse:.1%} (tolerance {ACCURACY_TOLERANCE:.0%})"
        )

    if not all_passed:
        raise RuntimeError(
            f"Simulation accuracy check failed: relative RMSE exceeded {ACCURACY_TOLERANCE:.0%} for one or more outputs."
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

    # Load model and evaluate with experimental data
    print("\n--- Experimental data accuracy check ---")
    try:
        check_evaluate_exp(config_dict, args.model)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    # Load model and evaluate with simulation data
    print("\n--- Simulation data accuracy check ---")
    try:
        check_evaluate_sim(config_dict, args.model)
    except Exception as e:
        print(f"[FAIL] {e}")
        sys.exit(1)

    print("\n[PASS] Model loaded and evaluated successfully.")
