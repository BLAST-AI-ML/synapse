#!/usr/bin/env python
# ruff: noqa: E402
## This script trains machine learning models (GP, NN, or ensemble_NN)
## using simulation and experimental data from MongoDB and saves trained models to MLflow
import time

import_start_time = time.time()

import argparse
import os
import torch
from botorch.models.transforms.input import AffineInputTransform
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, MaternKernel
import pymongo
import yaml
import mlflow
from lume_model.models import TorchModel
from lume_model.models.ensemble import NNEnsemble
from lume_model.models.gp_model import GPModel
from lume_model.variables import ScalarVariable
from lume_model.variables import DistributionVariable
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
from gpytorch.mlls import ExactMarginalLogLikelihood

sys.path.append(".")
from Neural_Net_Classes import CombinedNN, train_calibration

# measure the time it took to import everything
import_end_time = time.time()
elapsed_time = import_end_time - import_start_time
print(f"Imports took {elapsed_time:.1f} seconds.")

# Automatically select device for training of GP
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device selected: ", device)

start_time = time.time()


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="path to the configuration file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        help="Choose to train a model between GP, NN, or ensemble_NN",
        required=True,
    )
    parser.add_argument(
        "--test",
        help="Skip writing trained model to database (test mode)",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    config_file = args.config_file
    model_type = args.model
    test_mode = args.test
    print(
        f"Config file path: {config_file}, Model type: {model_type}, Test mode: {test_mode}"
    )
    if model_type not in ["NN", "ensemble_NN", "GP"]:
        raise ValueError(f"Invalid model type: {model_type}")
    return config_file, model_type, test_mode


def load_config(config_file):
    # Load configuration from the specified file path
    if not os.path.exists(config_file):
        raise RuntimeError(f"Configuration file not found: {config_file}")
    with open(config_file) as f:
        return yaml.safe_load(f.read())


def connect_to_db(config_dict):
    # Connect to the MongoDB database with read-only access
    db_host = config_dict["database"]["host"]
    db_name = config_dict["database"]["name"]
    db_auth = config_dict["database"]["auth"]
    db_username = config_dict["database"]["username_ro"]
    db_password_env = config_dict["database"]["password_ro_env"]
    db_password = os.getenv(db_password_env)
    if db_password is None:
        raise RuntimeError(f"Environment variable {db_password_env} must be set!")
    return pymongo.MongoClient(
        host=db_host,
        authSource=db_auth,
        username=db_username,
        password=db_password,
    )[db_name]


def normalize(df, input_names, input_normalization, output_names, output_normalization):
    # Apply normalization to the training data set
    norm_df = df.copy()
    norm_df[input_names] = input_normalization(torch.tensor(df[input_names].values))
    norm_df[output_names] = output_normalization(torch.tensor(df[output_names].values))
    return norm_df


def split_data(df, variables, model_type):
    if model_type == "GP":
        return (df[variables], None)
    else:
        # Split data into training and validation data with 80:20 ratio, selected randomly
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=None, shuffle=True
        )  # random_state will ensure the seed is different everytime, data will be shuffled randomly before splitting
        return (train_df[variables], val_df[variables])


def build_normalization(n_inputs, X_train, n_outputs, y_train):
    input_normalization = AffineInputTransform(
        n_inputs, coefficient=X_train.std(axis=0), offset=X_train.mean(axis=0)
    )
    # For output normalization, we need to handle potential NaN values
    y_mean = torch.nanmean(y_train, dim=0)
    y_std = torch.sqrt(torch.nanmean((y_train - y_mean) ** 2, dim=0))
    output_normalization = AffineInputTransform(
        n_outputs, coefficient=y_std, offset=y_mean
    )
    return input_normalization, output_normalization


def build_input_inferred_calibration(
    input_guess_calibration,
    input_normalization,
    input_inferred_normalizedcalibration,
    n_inputs,
):
    """
    Build input_inferred_calibration so that:
      [input_inferred_calibration, input_normalization]
    is equivalent to:
      [input_guess_calibration, input_normalization, input_inferred_normalizedcalibration]

    AffineInputTransform convention:
      T(x) = (x - offset) / coefficient
    """
    c_guess = input_guess_calibration.coefficient
    o_guess = input_guess_calibration.offset

    c_norm = input_normalization.coefficient
    o_norm = input_normalization.offset

    c_normcalibration = input_inferred_normalizedcalibration.coefficient
    o_normcalibration = input_inferred_normalizedcalibration.offset

    c_inferred = c_guess * c_normcalibration
    o_inferred = (
        o_guess
        + c_guess * o_norm
        + c_guess * c_norm * o_normcalibration
        - c_inferred * o_norm
    )

    input_inferred_calibration = AffineInputTransform(
        n_inputs,
        coefficient=c_inferred,
        offset=o_inferred,
    )

    # alpha_prime = 1.0 / c_inferred
    # beta_prime = o_inferred

    return input_inferred_calibration


def build_output_inferred_calibration(
    output_inferred_normalizedcalibration,
    output_normalization,
    output_guess_calibration,
    n_outputs,
):
    """
    Build output_inferred_calibration so that:
      [output_normalization, output_inferred_calibration]
    matches:
      [output_inferred_normalizedcalibration, output_normalization, output_guess_calibration]
    assuming lume-model applies output transformers via untransform().
    """
    c_normcalibration = output_inferred_normalizedcalibration.coefficient
    o_normcalibration = output_inferred_normalizedcalibration.offset

    c_norm = output_normalization.coefficient
    o_norm = output_normalization.offset

    c_guess = output_guess_calibration.coefficient
    o_guess = output_guess_calibration.offset

    c_inf = c_guess * c_normcalibration
    o_inf = (
        c_guess * c_norm * o_normcalibration
        + c_guess * (1.0 - c_normcalibration) * o_norm
        + o_guess
    )

    output_inferred_calibration = AffineInputTransform(
        n_outputs,
        coefficient=c_inf,
        offset=o_inf,
    )
    return output_inferred_calibration


def train_nn_ensemble(
    model_type,
    norm_df_train,
    norm_df_val,
    input_names,
    output_names,
    device,
):
    n_inputs = len(input_names)
    n_outputs = len(output_names)

    X_train = torch.tensor(
        norm_df_train[input_names].values,
        dtype=torch.float,
    ).to(device)
    y_train = torch.tensor(
        norm_df_train[output_names].values,
        dtype=torch.float,
    ).to(device)
    X_val = torch.tensor(
        norm_df_val[input_names].values,
        dtype=torch.float,
    ).to(device)
    y_val = torch.tensor(
        norm_df_val[output_names].values,
        dtype=torch.float,
    ).to(device)

    if model_type == "NN":
        num_models = 1
    elif model_type == "ensemble_NN":
        num_models = 10

    ensemble = []
    for i in range(num_models):
        model = CombinedNN(n_inputs, n_outputs, learning_rate=0.0001)
        model.to(device)  # moving to GPU
        NNmodel_start_time = time.time()
        model.train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            num_epochs=20000,
        )
        NNmodel_end_time = time.time()
        print(f"Model_{i + 1} trained in ", NNmodel_end_time - NNmodel_start_time)
        ensemble.append(model)

    return ensemble


def train_calibration_phase(
    model,
    model_type,
    norm_exp_df,
    input_names,
    output_names,
    device,
):
    """Phase 2: Train calibration layers on experimental data.

    Passes the frozen model to train_calibration(), which re-evaluates it at
    each iteration.

    Returns an AffineInputTransform representing the learned calibration.
    """
    exp_X = torch.tensor(
        norm_exp_df[input_names].values,
        dtype=torch.float,
    ).to(device)
    exp_y = torch.tensor(
        norm_exp_df[output_names].values,
        dtype=torch.float,
    ).to(device)

    # Build a predict callable that abstracts the NN vs GP difference
    if model_type == "GP":

        def predict_fn(x):
            return model.posterior(x.double()).mean.float().to(device)
    else:

        def predict_fn(x):
            return torch.stack([m.forward(x) for m in model]).mean(dim=0)

    # Train calibration
    input_cal_weight, input_cal_bias, output_cal_weight, output_cal_bias = (
        train_calibration(predict_fn, exp_X, exp_y, num_epochs=5000, lr=0.001)
    )

    # Build clibration transforms in normalized units
    input_inferred_normalizedcalibration = AffineInputTransform(
        len(input_names),
        coefficient=input_cal_weight.cpu(),
        offset=input_cal_bias.cpu(),
    )

    output_inferred_normalizedcalibration = AffineInputTransform(
        len(output_names),
        coefficient=output_cal_weight.cpu(),
        offset=output_cal_bias.cpu(),
    )
    return input_inferred_normalizedcalibration, output_inferred_normalizedcalibration


def build_lume_model(
    model,
    model_type,
    input_variables,
    output_variables,
    input_transformers,
    output_transformers,
):
    # Fix mismatch in name between the config file and the expected lume-model format
    for k in input_variables:
        input_variables[k]["default_value"] = input_variables[k]["default"]
        del input_variables[k]["default"]

    # Define lume-model input and output variables
    input_vars = [ScalarVariable(**input_variables[k]) for k in input_variables.keys()]
    output_vars = [
        ScalarVariable(**output_variables[k]) for k in output_variables.keys()
    ]
    if model_type in ["GP", "ensemble_NN"]:
        distribution_output_vars = [
            DistributionVariable(
                **output_variables[k], distribution_type="MultiVariateNormal"
            )
            for k in output_variables.keys()
        ]

    if model_type == "GP":
        return GPModel(
            model=model.cpu(),
            input_variables=input_vars,
            output_variables=distribution_output_vars,
            input_transformers=input_transformers,
            output_transformers=output_transformers,
        )
    else:
        # model is an ensemble list of NNs
        torch_models = []
        for model_nn in model:
            torch_models.append(
                TorchModel(
                    model=model_nn.cpu(),
                    input_variables=input_vars,
                    output_variables=output_vars,
                    input_transformers=input_transformers,
                    output_transformers=output_transformers,
                )
            )

        if model_type == "NN":
            # Return single NN
            return torch_models[0]
        else:
            # Return ensemble of NNs
            return NNEnsemble(
                models=torch_models,
                input_variables=input_vars,
                output_variables=distribution_output_vars,
            )


def train_gp(norm_df_train, input_names, output_names, device):
    # Create separate GP models for each output to handle NaN values in the training data
    gp_models = []

    for i, output_name in enumerate(output_names):
        print(f"Processing output {i + 1}/{len(output_names)}: {output_name}")

        # Get data where this output is not NaN
        output_data = norm_df_train[output_name].values
        valid_mask = torch.logical_not(torch.isnan(torch.tensor(output_data)))
        n_valid = torch.sum(valid_mask).item()
        print(f"Output {output_name}: {n_valid}/{len(output_data)} valid data points")

        # Prepare input and output data for this output
        X_valid = torch.tensor(
            norm_df_train[input_names].values[valid_mask], dtype=torch.float64
        )
        y_valid = torch.tensor(output_data[valid_mask], dtype=torch.float64).unsqueeze(
            -1
        )

        # SingleTaskGP for simulation data only
        gp_model = SingleTaskGP(
            X_valid,
            y_valid,
            covar_module=ScaleKernel(MaternKernel(nu=1.5)),
            outcome_transform=None,
        ).to(device)
        gp_models.append(gp_model)

    combined_gp = ModelListGP(*gp_models)
    print(f"ModelListGP created with {len(gp_models)} separate GP models")
    # Fit each separately
    for i, sub_gp in enumerate(gp_models):
        print(f"Training GP model {i + 1}/{len(gp_models)}...")
        mll = ExactMarginalLogLikelihood(sub_gp.likelihood, sub_gp)
        fit_gpytorch_mll(mll)

    return combined_gp


def enable_amsc_x_api_key(config_dict):
    """
    MLflow authentication helper for the AmSC MLflow server.
    Standard MLflow does not automatically inject custom headers like 'X-Api-Key'.
    This patches the http_request function to ensure every request to the server
    includes the AmSC API key.

    See https://gitlab.com/amsc2/ai-services/model-services/intro-to-mlflow-pytorch for more details.
    """
    import mlflow.utils.rest_utils as rest_utils

    mlflow_cfg = config_dict.get("mlflow") if config_dict is not None else None
    if not isinstance(mlflow_cfg, dict):
        raise KeyError(
            "Missing 'mlflow' configuration section required for AmSC MLFlow authentication."
        )

    api_key_env = mlflow_cfg.get("api_key_env")
    if not api_key_env:
        raise KeyError(
            "Missing 'api_key_env' in 'mlflow' configuration. "
            "Please specify the name of the environment variable containing the AmSC API key."
        )

    api_key = os.getenv(api_key_env)
    if api_key is None:
        raise KeyError(
            f"The environment variable '{api_key_env}' specified in 'mlflow.api_key_env' "
            "is not set. Please export it with the AmSC MLFlow API key."
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


def register_model_to_mlflow(model, model_type, experiment, config_dict):
    """Register the trained model to MLflow (tracking URI from config)."""
    tracking_uri = config_dict["mlflow"]["tracking_uri"]
    model_name = f"{experiment}_{model_type}"

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        model.register_to_mlflow(
            artifact_path=f"{model_name}_run",
            registered_model_name=model_name,
            code_paths=["Neural_Net_Classes.py"],
            log_model_dump=False,
        )
        print(f"Model registered to MLflow as {model_name}")
    except Exception as e:
        print(
            f"Failed to register model '{model_name}' to MLflow.\n"
            f"Tracking URI: {tracking_uri}\n"
            f"Experiment: {experiment}\n"
            f"Error: {e}"
        )
        raise RuntimeError(
            f"MLflow registration failed for model '{model_name}' "
            f"using tracking URI '{tracking_uri}' and experiment '{experiment}'."
        ) from e


# Main execution block
if __name__ == "__main__":
    # Parse command line arguments and load config
    experiment, model_type, test_mode = parse_arguments()
    config_dict = load_config(experiment)
    # Extract experiment name from config file
    experiment = config_dict["experiment"]
    print(f"Experiment: {experiment}")
    # Extract input and output variables from the config file
    input_variables = config_dict["inputs"]
    input_names = [v["name"] for v in input_variables.values()]
    output_variables = config_dict["outputs"]
    output_names = [v["name"] for v in output_variables.values()]

    # Extract experimental and simulation data from the database as pandas dataframe
    db = connect_to_db(config_dict)
    date_filter = config_dict.get("date_filter", {})
    df_exp = pd.DataFrame(db[experiment].find({"experiment_flag": 1, **date_filter}))
    df_sim = pd.DataFrame(db[experiment].find({"experiment_flag": 0}))

    # When using the AmSC MLflow: inject the X-Api-Key into the requests to authenticate with the MLflow server
    # (See https://gitlab.com/amsc2/ai-services/model-services/intro-to-mlflow-pytorch)
    if (
        "mlflow" in config_dict
        and config_dict["mlflow"].get("tracking_uri")
        == "https://mlflow.american-science-cloud.org"
    ):
        enable_amsc_x_api_key(config_dict)

    # Build simulation variable name mappings and alpha/beta vectors
    simulation_calibration = config_dict.get("simulation_calibration", {})
    n_inputs = len(input_names)
    n_outputs = len(output_names)

    # Build sim variable names and per-dimension alpha/beta for inputs
    sim_input_names = []
    alpha_input_list = []
    beta_input_list = []
    for key in input_variables:
        if key in simulation_calibration:
            sim_input_names.append(simulation_calibration[key]["name"])
            alpha_input_list.append(simulation_calibration[key]["alpha_guess"])
            beta_input_list.append(simulation_calibration[key]["beta_guess"])
        else:
            sim_input_names.append(input_variables[key]["name"])
            alpha_input_list.append(1.0)
            beta_input_list.append(0.0)

    # Build sim variable names and per-dimension alpha/beta for outputs
    sim_output_names = []
    alpha_output_list = []
    beta_output_list = []
    for key in output_variables:
        if key in simulation_calibration:
            sim_output_names.append(simulation_calibration[key]["name"])
            alpha_output_list.append(simulation_calibration[key]["alpha_guess"])
            beta_output_list.append(simulation_calibration[key]["beta_guess"])
        else:
            sim_output_names.append(output_variables[key]["name"])
            alpha_output_list.append(1.0)
            beta_output_list.append(0.0)

    alpha_inputs = torch.tensor(alpha_input_list, dtype=torch.float)
    beta_inputs = torch.tensor(beta_input_list, dtype=torch.float)
    alpha_outputs = torch.tensor(alpha_output_list, dtype=torch.float)
    beta_outputs = torch.tensor(beta_output_list, dtype=torch.float)

    # Build exp-to-sim and sim-to-exp AffineInputTransforms
    # exp_to_sim: sim = alpha * (exp - beta), i.e. AffineInputTransform with
    #   coefficient=1/alpha, offset=beta  =>  (exp - beta) / (1/alpha) = alpha*(exp-beta)
    input_guess_calibration = AffineInputTransform(
        n_inputs, coefficient=1.0 / alpha_inputs, offset=beta_inputs
    )
    output_guess_calibration = AffineInputTransform(
        n_outputs, coefficient=1.0 / alpha_outputs, offset=beta_outputs
    )

    # Convert experimental data to simulation variable space
    if len(df_exp) > 0:
        df_exp[sim_input_names] = (
            input_guess_calibration(
                torch.tensor(df_exp[input_names].values, dtype=torch.float)
            )
            .detach()
            .numpy()
        )
        df_exp[sim_output_names] = (
            output_guess_calibration(
                torch.tensor(df_exp[output_names].values, dtype=torch.float)
            )
            .detach()
            .numpy()
        )

    # Build normalization transforms in simulation variable space
    sim_variables = sim_input_names + sim_output_names
    if len(df_exp) > 0:
        df_all = pd.concat((df_exp[sim_variables], df_sim[sim_variables]))
    else:
        df_all = df_sim[sim_variables]
    X_all = torch.tensor(df_all[sim_input_names].values, dtype=torch.float)
    y_all = torch.tensor(df_all[sim_output_names].values, dtype=torch.float)
    input_normalization, output_normalization = build_normalization(
        n_inputs, X_all, n_outputs, y_all
    )

    # Split simulation data for Phase 1
    df_sim_train, df_sim_val = split_data(df_sim, sim_variables, model_type)

    # Normalize data
    norm_sim_train = normalize(
        df_sim_train,
        sim_input_names,
        input_normalization,
        sim_output_names,
        output_normalization,
    )
    norm_exp = None
    if len(df_exp) > 0:
        norm_exp = normalize(
            df_exp,
            sim_input_names,
            input_normalization,
            sim_output_names,
            output_normalization,
        )
    if model_type != "GP":
        # Single NN and ensemble of NNs
        norm_sim_val = normalize(
            df_sim_val,
            sim_input_names,
            input_normalization,
            sim_output_names,
            output_normalization,
        )

    # Phase 1: Train model on simulation data
    print("Phase 1: Training model on simulation data")
    train_start_time = time.time()
    if model_type != "GP":
        trained_model = train_nn_ensemble(
            model_type,
            norm_sim_train,
            norm_sim_val,
            sim_input_names,
            sim_output_names,
            device,
        )
    else:
        trained_model = train_gp(
            norm_sim_train,
            sim_input_names,
            sim_output_names,
            device,
        )
    print("Phase 1: training complete")

    # Phase 2: Train calibration on experimental data
    if norm_exp is not None and len(norm_exp) > 0:
        print("Phase 2: Training calibration on experimental data")
        input_inferred_normalizedcalibration, output_inferred_normalizedcalibration = (
            train_calibration_phase(
                trained_model,
                model_type,
                norm_exp,
                sim_input_names,
                sim_output_names,
                device,
            )
        )

        # Build calibration transfroms in physical units
        input_inferred_calibration = build_input_inferred_calibration(
            input_guess_calibration,
            input_normalization,
            input_inferred_normalizedcalibration,
            n_inputs,
        )

        input_transformers = [
            input_inferred_calibration,
            input_normalization,
        ]

        output_inferred_calibration = build_output_inferred_calibration(
            output_inferred_normalizedcalibration,
            output_normalization,
            output_guess_calibration,
            n_outputs,
        )

        output_transformers = [
            output_normalization,
            output_inferred_calibration,
        ]
        print("Phase 2: Calibration training complete")
    else:
        input_transformers = [input_guess_calibration, input_normalization]
        output_transformers = [output_normalization, output_guess_calibration]
        print("Phase 2: No experimental data available, skipping calibration")

    # Build LUME model
    model = build_lume_model(
        trained_model,
        model_type,
        input_variables,
        output_variables,
        input_transformers,
        output_transformers,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    data_time = train_start_time - start_time
    train_time = end_time - train_start_time
    print(f"Data prep time taken: {data_time:.2f} seconds")
    print(f"Train time taken: {train_time:.2f} seconds")
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    if test_mode:
        print("Test mode enabled: Skipping writing trained model to MLflow")
    elif "mlflow" in config_dict and config_dict["mlflow"].get("tracking_uri"):
        register_model_to_mlflow(model, model_type, experiment, config_dict)
    else:
        print(
            f"No mlflow.tracking_uri in configuration file for {experiment}; model not registered. "
            "Add an mlflow section with tracking_uri to store models in MLflow."
        )
