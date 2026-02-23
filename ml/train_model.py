#!/usr/bin/env python
# ruff: noqa: E402
## This script trains machine learning models (GP, NN, or ensemble_NN)
## using simulation and experimental data from MongoDB and saves trained models back to the database
##
## Training is done in two phases:
##   Phase 1: Train the model (NN or GP) on simulation data only (in normalized space)
##   Phase 2: Train calibration layers on experimental data (in normalized space)
import time

import_start_time = time.time()

import copy
import tempfile
import argparse
import torch
from botorch.models.transforms.input import AffineInputTransform
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, MaternKernel
import pymongo
import os
import re
import yaml
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
    # Connect to the MongoDB database with read-write access
    db_host = config_dict["database"]["host"]
    db_name = config_dict["database"]["name"]
    db_auth = config_dict["database"]["auth"]
    db_username = config_dict["database"]["username_rw"]
    db_password_env = config_dict["database"]["password_rw_env"]
    # Look for the password in the profile file
    with open(os.path.join(os.getenv("HOME"), "db.profile")) as f:
        db_profile = f.read()
    match = re.search(f"{db_password_env}='([^']*)'", db_profile)
    if not match:
        raise RuntimeError(f"Environment variable {db_password_env} must be set")
    db_password = match.group(1)
    return pymongo.MongoClient(
        host=db_host,
        authSource=db_auth,
        username=db_username,
        password=db_password,
    )[db_name]


def normalize(df, input_names, input_transform, output_names, output_transform):
    # Apply normalization to the training data set
    norm_df = df.copy()
    norm_df[input_names] = input_transform(torch.tensor(df[input_names].values))
    norm_df[output_names] = output_transform(torch.tensor(df[output_names].values))
    return norm_df


def split_sim_data(df_sim, variables, model_type):
    """Split simulation data for Phase 1 training.
    For GP: use all sim data (no split needed).
    For NN/ensemble_NN: 80/20 train/val split.
    """
    if model_type == "GP":
        return (df_sim[variables], None)
    else:
        # Split sim data into training and validation data with 80:20 ratio, selected randomly
        sim_train_df, sim_val_df = train_test_split(
            df_sim, test_size=0.2, random_state=None, shuffle=True
        )  # random_state will ensure the seed is different everytime, data will be shuffled randomly before splitting
        return (sim_train_df[variables], sim_val_df[variables])


def build_transforms(n_inputs, X_train, n_outputs, y_train):
    input_transform = AffineInputTransform(
        n_inputs, coefficient=X_train.std(axis=0), offset=X_train.mean(axis=0)
    )
    y_mean = torch.nanmean(y_train, dim=0)
    y_std = torch.sqrt(torch.nanmean((y_train - y_mean) ** 2, dim=0))
    output_transform = AffineInputTransform(n_outputs, coefficient=y_std, offset=y_mean)
    return input_transform, output_transform


def train_nn_ensemble(
    model_type,
    norm_sim_train,
    norm_sim_val,
    input_names,
    output_names,
    device,
):
    """Phase 1: Train NN ensemble on simulation data only."""
    n_inputs = len(input_names)
    n_outputs = len(output_names)

    X_train = torch.tensor(
        norm_sim_train[input_names].values,
        dtype=torch.float,
    ).to(device)
    y_train = torch.tensor(
        norm_sim_train[output_names].values,
        dtype=torch.float,
    ).to(device)
    X_val = torch.tensor(
        norm_sim_val[input_names].values,
        dtype=torch.float,
    ).to(device)
    y_val = torch.tensor(
        norm_sim_val[output_names].values,
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
    gp_models,
    norm_exp_df,
    input_names,
    output_names,
    device,
):
    """Phase 2: Train calibration layers on experimental data.

    Pre-computes model predictions on experimental inputs, then trains
    per-output affine calibration parameters (weight, bias).

    Returns an AffineInputTransform representing the learned calibration.
    """
    n_outputs = len(output_names)
    exp_X = torch.tensor(
        norm_exp_df[input_names].values,
        dtype=torch.float,
    ).to(device)
    exp_y = torch.tensor(
        norm_exp_df[output_names].values,
        dtype=torch.float,
    ).to(device)

    # Pre-compute base model predictions
    with torch.no_grad():
        if model_type == "GP":
            gp_preds = []
            for i, sub_gp in enumerate(gp_models):
                sub_gp.eval()
                posterior = sub_gp.posterior(exp_X.double())
                gp_preds.append(posterior.mean.squeeze(-1))
            base_predictions = torch.stack(gp_preds, dim=-1).float().to(device)
        else:
            model.eval()
            base_predictions = model(exp_X)

    print(f"Phase 2: Training calibration on {len(exp_X)} experimental data points")
    cal_weight, cal_bias = train_calibration(
        base_predictions, exp_y, n_outputs, num_epochs=5000, lr=0.001
    )

    calibration_transform = AffineInputTransform(
        n_outputs,
        coefficient=cal_weight.cpu(),
        offset=cal_bias.cpu(),
    )
    return calibration_transform


def build_torch_model_from_nn(
    ensemble,
    model_type,
    input_variables,
    output_variables,
    input_transform,
    output_transform,
    calibration_transform,
    output_names,
):
    torch_models = []

    # Fix mismatch in name between the config file and the expected lume-model format
    iv = copy.deepcopy(input_variables)
    for k in iv:
        iv[k]["default_value"] = iv[k].pop("default", iv[k].get("default_value"))

    for model_nn in ensemble:
        output_transformers = [output_transform]
        if calibration_transform is not None:
            output_transformers = [calibration_transform, output_transform]

        torch_models.append(
            TorchModel(
                model=model_nn,
                input_variables=[ScalarVariable(**iv[k]) for k in iv.keys()],
                output_variables=[
                    ScalarVariable(**output_variables[k])
                    for k in output_variables.keys()
                ],
                input_transformers=[input_transform],
                output_transformers=output_transformers,
            )
        )

    if model_type == "NN":
        return torch_models[0]
    else:
        return NNEnsemble(
            models=torch_models,
            input_variables=[ScalarVariable(**iv[k]) for k in iv.keys()],
            output_variables=[
                DistributionVariable(**output_variables[k])
                for k in output_variables.keys()
            ],
        )


def train_gp(norm_df_train, input_names, output_names, device):
    """Phase 1: Train GP on simulation data only.
    Returns the combined ModelListGP and the list of individual GP models.
    """
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
    GP_start_time = time.time()
    for i, sub_gp in enumerate(gp_models):
        print(f"Training GP model {i + 1}/{len(gp_models)}...")
        mll = ExactMarginalLogLikelihood(sub_gp.likelihood, sub_gp)
        fit_gpytorch_mll(mll)
    GP_end_time = time.time()
    print(f"All GP models training time: {GP_end_time - GP_start_time:.2f} seconds")

    return combined_gp, gp_models


def build_lume_gp_model(
    combined_gp,
    gp_models,
    input_variables,
    input_transform,
    output_transform,
    calibration_transform,
    output_names,
):
    """Build a lume-model GPModel from already-trained GP models."""
    iv = copy.deepcopy(input_variables)
    for k in iv:
        iv[k]["default_value"] = iv[k].pop("default", iv[k].get("default_value"))

    output_variables_list = [
        DistributionVariable(
            name=f"{name}_sim_task", distribution_type="MultiVariateNormal"
        )
        for name in output_names
    ]

    output_transformers = [output_transform]
    if calibration_transform is not None:
        output_transformers = [calibration_transform, output_transform]

    return GPModel(
        model=combined_gp.cpu(),
        input_variables=[ScalarVariable(**iv[k]) for k in iv.keys()],
        output_variables=output_variables_list,
        input_transformers=[input_transform],
        output_transformers=output_transformers,
    )


def write_model(model, model_type, experiment, db):
    with tempfile.TemporaryDirectory() as temp_dir:
        if model_type != "GP":
            model.dump(file=os.path.join(temp_dir, experiment + ".yml"), save_jit=True)
        else:
            model.dump(
                file=os.path.join(temp_dir, experiment + ".yml"), save_models=True
            )
        # Upload the model to the database
        # - Load the files that were just created into a dictionary
        with open(os.path.join(temp_dir, experiment + ".yml")) as f:
            yaml_file_content = f.read()
        document = {
            "experiment": experiment,
            "model_type": model_type,
            "yaml_file_content": yaml_file_content,
        }
        # Extract list of files to upload
        files_to_upload = []
        if model_type == "ensemble_NN":
            models_info = yaml.safe_load(yaml_file_content)
            for model in models_info["models"]:
                yaml_file_name = model.replace("_model.jit", ".yml")
                files_to_upload.append(yaml_file_name)
                with open(os.path.join(temp_dir, yaml_file_name)) as f:
                    model_info = yaml.safe_load(f.read())
                # Extract files to upload
                files_to_upload += (
                    [model_info["model"]]
                    + model_info["input_transformers"]
                    + model_info["output_transformers"]
                )
        else:
            # Extract files to upload
            model_info = yaml.safe_load(yaml_file_content)
            files_to_upload += (
                [model_info["model"]]
                + model_info["input_transformers"]
                + model_info["output_transformers"]
            )
        # Upload all the files that define the model(s)
        for filename in files_to_upload:
            with open(os.path.join(temp_dir, filename), "rb") as f:
                document[filename] = f.read()
        # - Check whether there is already a model in the database
        query = {"experiment": experiment, "model_type": model_type}
        count = db["models"].count_documents(query)
        # - Upload/replace the model in the database
        if count > 1:
            print(
                f"Multiple models found for experiment: {experiment} and model type: {model_type}! Removing them."
            )
            db["models"].delete_many(query)
        elif count == 1:
            print(
                f"Model already exists for experiment: {experiment} and model type: {model_type}! Removing it."
            )
            db["models"].delete_one(query)
        print("Uploading new model to database")
        db["models"].insert_one(document)
        print("Model uploaded to database")


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

    # Apply simulation calibration to the simulation data
    if "simulation_calibration" in config_dict:
        simulation_calibration = config_dict["simulation_calibration"]
    else:
        simulation_calibration = {}
    for value in simulation_calibration.values():
        sim_name = value["name"]
        exp_name = value["depends_on"]
        df_sim[exp_name] = df_sim[sim_name] / value["alpha_guess"] + value["beta_guess"]

    # Apply normalization to the training data
    # Build normalization transforms from ALL data (sim + exp) for consistent ranges
    variables = input_names + output_names
    if len(df_exp) > 0:
        df_all = pd.concat((df_exp[variables], df_sim[variables]))
    else:
        df_all = df_sim[variables]
    X_all = torch.tensor(df_all[input_names].values, dtype=torch.float)
    y_all = torch.tensor(df_all[output_names].values, dtype=torch.float)
    input_transform, output_transform = build_transforms(
        len(input_names), X_all, len(output_names), y_all
    )

    # Split simulation data for Phase 1
    sim_variables = input_names + output_names
    df_sim_train, df_sim_val = split_sim_data(df_sim, sim_variables, model_type)

    # Normalize simulation data
    norm_sim_train = normalize(
        df_sim_train, input_names, input_transform, output_names, output_transform
    )

    # Normalize experimental data for Phase 2 (if available)
    norm_exp = None
    if len(df_exp) > 0:
        norm_exp = normalize(
            df_exp, input_names, input_transform, output_names, output_transform
        )

    model = None
    calibration_transform = None

    ######################################################
    # Phase 1: Train model on simulation data only
    ######################################################
    if model_type != "GP":
        norm_sim_val = normalize(
            df_sim_val, input_names, input_transform, output_names, output_transform
        )
        print("Phase 1: Training NN on simulation data")
        NN_start_time = time.time()
        ensemble = train_nn_ensemble(
            model_type,
            norm_sim_train,
            norm_sim_val,
            input_names,
            output_names,
            device,
        )
        print("Phase 1: NN training complete")

        ######################################################
        # Phase 2: Train calibration on experimental data
        ######################################################
        if norm_exp is not None and len(norm_exp) > 0:
            print("Phase 2: Training calibration on experimental data")
            calibration_transform = train_calibration_phase(
                ensemble[0] if model_type == "NN" else ensemble[0],
                model_type,
                None,
                norm_exp,
                input_names,
                output_names,
                device,
            )
            print("Phase 2: Calibration training complete")
        else:
            print("Phase 2: No experimental data available, skipping calibration")

        model = build_torch_model_from_nn(
            ensemble,
            model_type,
            input_variables,
            output_variables,
            input_transform,
            output_transform,
            calibration_transform,
            output_names,
        )
        end_time = time.time()

        elapsed_time = end_time - start_time
        data_time = NN_start_time - start_time
        NN_time = end_time - NN_start_time
        print(f"Total time taken: {elapsed_time:.2f} seconds")
        print(f"Data prep time taken: {data_time:.2f} seconds")
        print(f"NN time taken: {NN_time:.2f} seconds")

    ###############################################################
    # GP: Phase 1 (training) + Phase 2 (calibration)
    ###############################################################
    else:
        print("Phase 1: Training GP on simulation data")
        combined_gp, gp_models = train_gp(
            norm_sim_train,
            input_names,
            output_names,
            device,
        )
        print("Phase 1: GP training complete")

        # Phase 2: Train calibration
        if norm_exp is not None and len(norm_exp) > 0:
            print("Phase 2: Training calibration on experimental data")
            calibration_transform = train_calibration_phase(
                None,
                "GP",
                gp_models,
                norm_exp,
                input_names,
                output_names,
                device,
            )
            print("Phase 2: Calibration training complete")
        else:
            print("Phase 2: No experimental data available, skipping calibration")

        model = build_lume_gp_model(
            combined_gp,
            gp_models,
            input_variables,
            input_transform,
            output_transform,
            calibration_transform,
            output_names,
        )

    if not test_mode:
        write_model(model, model_type, experiment, db)
    else:
        print("Test mode enabled: Skipping writing trained model to database")
