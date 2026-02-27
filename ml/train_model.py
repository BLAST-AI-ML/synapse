#!/usr/bin/env python
# ruff: noqa: E402
## This script trains machine learning models (GP, NN, or ensemble_NN)
## using simulation and experimental data from MongoDB and saves trained models back to the database
import time

import_start_time = time.time()

import tempfile
import argparse
import torch
from botorch.models.transforms.input import AffineInputTransform
from botorch.models import MultiTaskGP, SingleTaskGP, ModelListGP
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
from Neural_Net_Classes import CombinedNN as CombinedNN

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


def build_transforms(n_inputs, X_train, n_outputs, y_train):
    input_transform = AffineInputTransform(
        n_inputs, coefficient=X_train.std(axis=0), offset=X_train.mean(axis=0)
    )
    # For output normalization, we need to handle potential NaN values
    y_mean = torch.nanmean(y_train, dim=0)
    y_std = torch.sqrt(torch.nanmean((y_train - y_mean) ** 2, dim=0))
    output_transform = AffineInputTransform(n_outputs, coefficient=y_std, offset=y_mean)
    return input_transform, output_transform


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
        train_on_experiments = 1
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
            train_on_expt
        )
        NNmodel_end_time = time.time()
        print(f"Model_{i + 1} trained in ", NNmodel_end_time - NNmodel_start_time)
        ensemble.append(model)

    return ensemble


def build_torch_model_from_nn(
    ensemble,
    model_type,
    input_variables,
    output_variables,
    input_transform,
    output_transform,
    output_names,
):
    torch_models = []

    for model_nn in ensemble:
        calibration_transform = AffineInputTransform(
            len(output_names),
            coefficient=model_nn.sim_to_exp_calibration_weight.clone().detach().cpu(),
            offset=model_nn.sim_to_exp_calibration_bias.clone().detach().cpu(),
        )

        # Fix mismatch in name between the config file and the expected lume-model format
        for k in input_variables:
            input_variables[k]["default_value"] = input_variables[k]["default"]

        torch_models.append(
            TorchModel(
                model=model_nn,
                input_variables=[
                    ScalarVariable(**input_variables[k]) for k in input_variables.keys()
                ],
                output_variables=[
                    ScalarVariable(**output_variables[k])
                    for k in output_variables.keys()
                ],
                input_transformers=[input_transform],
                output_transformers=[
                    calibration_transform,
                    output_transform,
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
            input_variables=[
                ScalarVariable(**input_variables[k]) for k in input_variables.keys()
            ],
            output_variables=[
                DistributionVariable(**output_variables[k])
                for k in output_variables.keys()
            ],
        )


def train_gp(
    norm_df_train, input_names, output_names, input_transform, output_transform, device
):
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

        # Create GP model based on whether we have experimental data
        if (
            False
        ):  # len(df_exp) > 0: # Temporarily deactivate MultiTaskGP for simplicity
            # MultiTaskGP for experimental vs simulation data
            exp_flag_valid = torch.tensor(
                norm_df_train[["experiment_flag"]].values[valid_mask],
                dtype=torch.float64,
            )
            X_with_task = torch.cat([exp_flag_valid, X_valid], dim=1)

            gp_model = MultiTaskGP(
                X_with_task,
                y_valid,
                task_feature=0,
                covar_module=ScaleKernel(MaternKernel(nu=1.5)),
                outcome_transform=None,
            ).to(device)

        else:
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

    # Fix mismatch in name between the config file and the expected lume-model format
    for k in input_variables:
        input_variables[k]["default_value"] = input_variables[k]["default"]
        del input_variables[k]["default"]

    if False:  # len(df_exp) > 0: # Temporarily deactivate MultiTaskGP for simplicity
        output_variables = [
            DistributionVariable(
                name=f"{name}_{suffix}", distribution_type="MultiVariateNormal"
            )
            for name in output_names
            for suffix in ["sim_task", "exp_task"]
        ]
    else:
        output_variables = [
            DistributionVariable(
                name=f"{name}_{suffix}", distribution_type="MultiVariateNormal"
            )
            for name in output_names
            for suffix in ["sim_task"]
        ]

    return GPModel(
        model=combined_gp.cpu(),
        input_variables=[
            ScalarVariable(**input_variables[k]) for k in input_variables.keys()
        ],
        output_variables=output_variables,
        input_transformers=[input_transform],
        output_transformers=[output_transform],
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

    # Concatenate experimental and simulation data for training and validation
    variables = input_names + output_names + ["experiment_flag"]
    df_train, df_val = split_data(df_exp, df_sim, variables, model_type)

    # Apply normalization to the training data
    X_train = torch.tensor(df_train[input_names].values, dtype=torch.float)
    y_train = torch.tensor(df_train[output_names].values, dtype=torch.float)
    input_transform, output_transform = build_transforms(
        len(input_names), X_train, len(output_names), y_train
    )
    norm_df_train = normalize(
        df_train, input_names, input_transform, output_names, output_transform
    )

    model = None
    ######################################################
    # Neural Net and Ensemble Creation and training
    ######################################################
    if model_type != "GP":
        norm_df_val = normalize(
            df_val, input_names, input_transform, output_names, output_transform
        )
        print("training started")
        NN_start_time = time.time()
        ensemble = train_nn_ensemble(
            model_type,
            norm_df_train,
            norm_df_val,
            input_names,
            output_names,
            device,
        )
        print("training ended")

        model = build_torch_model_from_nn(
            ensemble,
            model_type,
            input_variables,
            output_variables,
            input_transform,
            output_transform,
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
    # Gaussian Process Creation and training
    ###############################################################
    else:
        model = train_gp(
            norm_df_train,
            input_names,
            output_names,
            input_transform,
            output_transform,
            device,
        )

    if not test_mode:
        write_model(model, model_type, experiment, db)
    else:
        print("Test mode enabled: Skipping writing trained model to database")
