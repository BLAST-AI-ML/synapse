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
from Neural_Net_Classes import CombinedNN

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


def split_data(df_exp, df_sim, variables, model_type):
    if model_type == "GP":
        if len(df_exp) > 0:
            return (pd.concat((df_exp[variables], df_sim[variables])), None)
        else:
            return (df_sim[variables], None)
    else:
        # Split exp and sim data into training and validation data with 80:20 ratio, selected randomly
        sim_train_df, sim_val_df = train_test_split(
            df_sim, test_size=0.2, random_state=None, shuffle=True
        )  # random_state will ensure the seed is different everytime, data will be shuffled randomly before splitting
        if len(df_exp) > 0:
            exp_train_df, exp_val_df = train_test_split(
                df_exp, test_size=0.2, random_state=None, shuffle=True
            )  # 20% of the data will go in validation test, no fixing the
            return (
                pd.concat((exp_train_df[variables], sim_train_df[variables])),
                pd.concat((exp_val_df[variables], sim_val_df[variables])),
            )
        else:
            return (sim_train_df[variables], sim_val_df[variables])


def build_normalizations(n_inputs, X_train, n_outputs, y_train):
    input_normalization = AffineInputTransform(
        n_inputs, coefficient=X_train.std(axis=0), offset=X_train.mean(axis=0)
    )
    # For output normalization, we need to handle potential NaN values
    y_mean = torch.nanmean(y_train, dim=0)
    y_std = torch.sqrt(torch.nanmean((y_train - y_mean) ** 2, dim=0))
    output_normalization = AffineInputTransform(n_outputs, coefficient=y_std, offset=y_mean)
    return input_normalization, output_normalization


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

    exp_X_train = torch.tensor(
        norm_df_train[norm_df_train.experiment_flag == 1][input_names].values,
        dtype=torch.float,
    ).to(device)
    exp_y_train = torch.tensor(
        norm_df_train[norm_df_train.experiment_flag == 1][output_names].values,
        dtype=torch.float,
    ).to(device)
    sim_X_train = torch.tensor(
        norm_df_train[norm_df_train.experiment_flag == 0][input_names].values,
        dtype=torch.float,
    ).to(device)
    sim_y_train = torch.tensor(
        norm_df_train[norm_df_train.experiment_flag == 0][output_names].values,
        dtype=torch.float,
    ).to(device)
    exp_X_val = torch.tensor(
        norm_df_val[norm_df_val.experiment_flag == 1][input_names].values,
        dtype=torch.float,
    ).to(device)
    exp_y_val = torch.tensor(
        norm_df_val[norm_df_val.experiment_flag == 1][output_names].values,
        dtype=torch.float,
    ).to(device)
    sim_X_val = torch.tensor(
        norm_df_val[norm_df_val.experiment_flag == 0][input_names].values,
        dtype=torch.float,
    ).to(device)
    sim_y_val = torch.tensor(
        norm_df_val[norm_df_val.experiment_flag == 0][output_names].values,
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
            sim_X_train,
            sim_y_train,
            exp_X_train,
            exp_y_train,
            sim_X_val,
            sim_y_val,
            exp_X_val,
            exp_y_val,
            num_epochs=20000,
        )
        NNmodel_end_time = time.time()
        print(f"Model_{i + 1} trained in ", NNmodel_end_time - NNmodel_start_time)
        ensemble.append(model)

    return ensemble


def build_lume_model(
    model,
    model_type,
    input_variables,
    output_variables,
    input_normalization,
    output_normalization,
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

    # Create lume-model objects
    if model_type == "GP":
        return GPModel(
            model=model.cpu(),
            input_variables=input_vars,
            output_variables=distribution_output_vars,
            input_transformers=[input_normalization],
            output_transformers=[output_normalization],
        )
    else:
        # model is an ensemble list of NNs
        torch_models = []
        for model_nn in model:
            calibration_transform = AffineInputTransform(
                len(output_vars),
                coefficient=model_nn.sim_to_exp_calibration_weight.clone()
                .detach()
                .cpu(),
                offset=model_nn.sim_to_exp_calibration_bias.clone().detach().cpu(),
            )

            torch_models.append(
                TorchModel(
                    model=model_nn.cpu(),
                    input_variables=input_vars,
                    output_variables=output_vars,
                    input_transformers=[input_normalization],
                    output_transformers=[
                        calibration_transform,
                        output_normalization,
                    ],  # saving calibration before normalization
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

        # Create GP model
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
    n_outputs = len(output_names)

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

    # Apply simulation calibration to the simulation data
    if "simulation_calibration" in config_dict:
        simulation_calibration = config_dict["simulation_calibration"]
    else:
        simulation_calibration = {}
    for value in simulation_calibration.values():
        sim_name = value["name"]
        exp_name = value["depends_on"]
        df_sim[exp_name] = df_sim[sim_name] / value["alpha_guess"] + value["beta_guess"]

    # Concatenate experimental and simulation data for training and validation
    variables = input_names + output_names + ["experiment_flag"]
    df_train, df_val = split_data(df_exp, df_sim, variables, model_type)

    # Apply normalization to the training data
    X_train = torch.tensor(df_train[input_names].values, dtype=torch.float)
    y_train = torch.tensor(df_train[output_names].values, dtype=torch.float)
    input_normalization, output_normalization = build_normalizations(
        len(input_names), X_train, len(output_names), y_train
    )
    norm_df_train = normalize(
        df_train, input_names, input_normalization, output_names, output_normalization
    )
    if model_type != "GP":
        norm_df_val = normalize(
            df_val, input_names, input_normalization, output_names, output_normalization
        )

    print("training started")
    train_start_time = time.time()
    ######################################################
    # Neural Net and Ensemble Creation and training
    ######################################################
    if model_type != "GP":
        trained_model = train_nn_ensemble(
            model_type,
            norm_df_train,
            norm_df_val,
            input_names,
            output_names,
            device,
        )
    ###############################################################
    # Gaussian Process Creation and training
    ###############################################################
    else:
        trained_model = train_gp(
            norm_df_train,
            input_names,
            output_names,
            device,
        )

    print("training ended")

    end_time = time.time()

    elapsed_time = end_time - start_time
    data_time = train_start_time - start_time
    train_time = end_time - train_start_time
    print(f"Data prep time taken: {data_time:.2f} seconds")
    print(f"Train time taken: {train_time:.2f} seconds")
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    model = build_lume_model(
        trained_model,
        model_type,
        input_variables,
        output_variables,
        input_normalization,
        output_normalization,
    )

    if test_mode:
        print("Test mode enabled: Skipping writing trained model to MLflow")
    elif "mlflow" in config_dict and config_dict["mlflow"].get("tracking_uri"):
        register_model_to_mlflow(model, model_type, experiment, config_dict)
    else:
        print(
            f"No mlflow.tracking_uri in configuration file for {experiment}; model not registered. "
            "Add an mlflow section with tracking_uri to store models in MLflow."
        )
