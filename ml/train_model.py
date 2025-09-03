#!/usr/bin/env python
## This notebook includes simulation and experimental data
## in a database using PyMongo
## Author : Revathi Jambunathan, Axel Huebl
## Date : January, 2025
import time
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

# Automatically select device for training of GP
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device selected: ', device)

############################################
# Get command line arguments
############################################
# define parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment",
    help="name/tag of the experiment",
    type=str,
    required=True,
)
parser.add_argument(
    "--model",
    help="Choose to train a model between GP, NN, or ensemble_NN",
    required=True
)
args = parser.parse_args()
experiment = args.experiment
model_type = args.model
print(f"Experiment: {experiment}, Model type: {model_type}")
start_time = time.time()
if model_type not in ['NN', 'ensemble_NN', 'GP']:
    raise ValueError(f"Invalid model type: {model_type}")

###############################################
# Open credential file for database
###############################################
with open(os.path.join(os.getenv('HOME'), 'db.profile')) as f:
    db_profile = f.read()

# Connect to the MongoDB database with read-only access
db = pymongo.MongoClient(
    host="mongodb05.nersc.gov",
    username="bella_sf_admin",
    password=re.findall('SF_DB_ADMIN_PASSWORD=(.+)', db_profile)[0],
    authSource="bella_sf")["bella_sf"]

# Extract configurations of experiments & models
yaml_dict = None
current_file_directory = os.path.dirname(os.path.abspath(__file__))
config_dir_locations = [current_file_directory, "./", "../dashboard/config/"]
for config_dir in config_dir_locations:
    file_path = config_dir + "variables.yml"
    if os.path.exists(file_path):
        with open(file_path) as f:
            yaml_dict = yaml.safe_load( f.read() )
        break
if yaml_dict is None:
    raise RuntimeError("File variables.yml not found.")

input_variables = yaml_dict[experiment]["input_variables"]
input_names = [ v['name'] for v in input_variables.values() ]
output_variables = yaml_dict[experiment]["output_variables"]
output_names = [ v['name'] for v in output_variables.values() ]
n_outputs = len(output_names)
# Extract data from the database as pandas dataframe
collection = db[experiment]
df_exp = pd.DataFrame(db[experiment].find({"experiment_flag": 1}))
df_sim = pd.DataFrame(db[experiment].find({"experiment_flag": 0}))
# Apply calibration to the simulation results
if "simulation_calibration" in yaml_dict[experiment]:
    simulation_calibration = yaml_dict[experiment]["simulation_calibration"]
else:
    simulation_calibration = {}
for _, value in simulation_calibration.items():
    sim_name = value["name"]
    exp_name = value["depends_on"]
    df_sim[exp_name] = df_sim[sim_name] / value["alpha"] + value["beta"]

# Concatenate experimental and simulation data for training and validation
variables = input_names + output_names + ['experiment_flag']
if model_type != 'GP':
    #Split exp and sim data into training and validation data with 80:20 ratio, selected randomly
    sim_train_df, sim_val_df = train_test_split(df_sim, test_size=0.2, random_state=None, shuffle=True)#random_state will ensure the seed is different everytime, data will be shuffled randomly before splitting
    if len(df_exp) > 0:
        exp_train_df, exp_val_df = train_test_split(df_exp, test_size=0.2, random_state=None, shuffle=True)# 20% of the data will go in validation test, no fixing the
        df_train = pd.concat( (exp_train_df[variables], sim_train_df[variables]) )
        df_val = pd.concat( (exp_val_df[variables], sim_val_df[variables]) )
    else:
        df_train = sim_train_df[variables]
        df_val = sim_val_df[variables]
else:
    # No split: all the data is training data
    if len(df_exp) > 0:
        df_train = pd.concat( (df_exp[variables], df_sim[variables]) )
    else:
        df_train = df_sim[variables]

# Normalize with Affine Input Transformer
# Define the input and output normalizations
X_train = torch.tensor( df_train[ input_names ].values, dtype=torch.float )
input_transform = AffineInputTransform(
    len(input_names),
    coefficient=X_train.std(axis=0),
    offset=X_train.mean(axis=0)
)
# For output normalization, we need to handle potential NaN values
y_train = torch.tensor( df_train[ output_names ].values, dtype=torch.float )
y_mean = torch.nanmean(y_train, dim=0)
y_std = torch.sqrt( torch.nanmean( (y_train-y_mean)**2, dim=0) )
output_transform = AffineInputTransform(
    n_outputs,
    coefficient=y_std,
    offset=y_mean
)

# Apply normalization to the training data set
norm_df_train = df_train.copy()
norm_df_train[input_names] = input_transform( torch.tensor( df_train[input_names].values ) )
norm_df_train[output_names] = output_transform( torch.tensor( df_train[output_names].values ) )

norm_expt_inputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==1][input_names].values, dtype=torch.float)
norm_expt_outputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==1][output_names].values, dtype=torch.float)
norm_sim_inputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==0][input_names].values, dtype=torch.float)
norm_sim_outputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==0][output_names].values, dtype=torch.float)

model = None
######################################################
# Neural Net and Ensemble Creation and training
######################################################
if model_type != 'GP':
    # Saving the Lume Model - TO do for combined NN
    ##############################
    #Early Stopping and validation
    ##############################
    # Apply normalization to the validation data set
    norm_df_val = df_val.copy()
    norm_df_val[input_names] = input_transform( torch.tensor( df_val[input_names].values ) )
    norm_df_val[output_names] = output_transform( torch.tensor( df_val[output_names].values ) )

    norm_expt_inputs_val = torch.tensor( norm_df_val[norm_df_val.experiment_flag==1][input_names].values, dtype=torch.float)
    norm_expt_outputs_val = torch.tensor( norm_df_val[norm_df_val.experiment_flag==1][output_names].values, dtype=torch.float)
    norm_sim_inputs_val = torch.tensor( norm_df_val[norm_df_val.experiment_flag==0][input_names].values, dtype=torch.float)
    norm_sim_outputs_val = torch.tensor( norm_df_val[norm_df_val.experiment_flag==0][output_names].values, dtype=torch.float)

    print("training started")

    NN_start_time = time.time()
    if model_type == 'NN':
        num_models = 1
    elif model_type == 'ensemble_NN':
        num_models = 10

    ensemble = []
    for i in range(num_models):
        model = CombinedNN(len(input_names), n_outputs, learning_rate=0.0001)
        model.to(device) # moving to GPU
        NNmodel_start_time = time.time()
        model.train_model(
            norm_sim_inputs_train.to(device), norm_sim_outputs_train.to(device),
            norm_expt_inputs_train.to(device), norm_expt_outputs_train.to(device),
            norm_sim_inputs_val.to(device), norm_sim_outputs_val.to(device),
            norm_expt_inputs_val.to(device), norm_expt_outputs_val.to(device),
            num_epochs=20000)
        NNmodel_end_time = time.time()
        print(f'Model_{i+1} trained in ', NNmodel_end_time - NNmodel_start_time)
        ensemble.append(model)


    torch_models = []
    for model_nn in ensemble:
        calibration_transform = AffineInputTransform(
            n_outputs,
            coefficient=model_nn.sim_to_exp_calibration_weight.clone().detach().cpu(),
            offset=model_nn.sim_to_exp_calibration_bias.clone().detach().cpu() )

        # Fix mismatch in name between the config file and the expected lume-model format
        for k in input_variables:
            print(input_variables[k])
            input_variables[k]['default_value'] = input_variables[k]['default']

        torch_model = TorchModel(
            model=model_nn,
            input_variables=[ ScalarVariable(**input_variables[k]) for k in input_variables.keys() ],
            output_variables=[ ScalarVariable(**output_variables[k]) for k in output_variables.keys() ],
            input_transformers=[input_transform],
            output_transformers=[calibration_transform,output_transform] # saving calibration before normalization
        )
        if num_models == 1:
            #Save single NN and break
            model = torch_model
            end_time = time.time()
            break
        torch_models.append(torch_model)

    #Save Ensemble
    if num_models > 1:
        ensemble = NNEnsemble(
        models=torch_models,
        input_variables=[ ScalarVariable(**input_variables[k]) for k in input_variables.keys() ],
        output_variables=[ DistributionVariable(**output_variables[k]) for k in output_variables.keys() ]
        )
        model = ensemble
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
    # Create separate GP models for each output to handle NaN values

    gp_models = []
    print(f"Creating separate GP models for {n_outputs} outputs...")

    for i, output_name in enumerate(output_names):
        print(f"Processing output {i+1}/{n_outputs}: {output_name}")

        # Get data where this output is not NaN
        output_data = norm_df_train[output_name].values
        valid_mask = torch.logical_not( torch.isnan(torch.tensor(output_data)) )
        n_valid = torch.sum(valid_mask).item()
        print(f"Output {output_name}: {n_valid}/{len(output_data)} valid data points")

        # Prepare input and output data for this output
        X_valid = torch.tensor(norm_df_train[input_names].values[valid_mask], dtype=torch.float64)
        y_valid = torch.tensor(output_data[valid_mask], dtype=torch.float64).unsqueeze(-1)

        # Create GP model based on whether we have experimental data
        if False: # len(df_exp) > 0: # Temporarily deactivate MultiTaskGP for simplicity
            # MultiTaskGP for experimental vs simulation data
            exp_flag_valid = torch.tensor(norm_df_train[['experiment_flag']].values[valid_mask], dtype=torch.float64)
            X_with_task = torch.cat([exp_flag_valid, X_valid], dim=1)

            gp_model = MultiTaskGP(
                X_with_task, y_valid,
                task_feature=0,
                covar_module=ScaleKernel(MaternKernel(nu=1.5)),
                outcome_transform=None,
            ).to(device)

        else:
            # SingleTaskGP for simulation data only
            gp_model = SingleTaskGP(
                X_valid, y_valid,
                covar_module=ScaleKernel(MaternKernel(nu=1.5)),
                outcome_transform=None,
            ).to(device)

        gp_models.append(gp_model)

    # Combine the models in a ModelListGP
    gp_model = ModelListGP(*gp_models)
    print(f"ModelListGP created with {len(gp_models)} separate GP models")
    # Fit each separately
    GP_start_time = time.time()
    for i, model in enumerate(gp_models):
        print(f"Training GP model {i+1}/{len(gp_models)}...")
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
    GP_end_time = time.time()
    print(f"All GP models training time: {GP_end_time - GP_start_time:.2f} seconds")

    # Fix mismatch in name between the config file and the expected lume-model format
    for k in input_variables:
        print(input_variables[k])
        input_variables[k]['default_value'] = input_variables[k]['default']
        del input_variables[k]['default']

    input_variables = [ ScalarVariable(**input_variables[k]) for k in input_variables.keys() ]

    if False: #len(df_exp) > 0: # Temporarily deactivate MultiTaskGP for simplicity
        output_variables = [
            DistributionVariable(name=f"{name}_{suffix}", distribution_type="MultiVariateNormal")
            for name in output_names
            for suffix in ["sim_task", "exp_task"]
        ]
    else:
        output_variables = [
            DistributionVariable(name=f"{name}_{suffix}", distribution_type="MultiVariateNormal")
            for name in output_names
            for suffix in ["sim_task"]
        ]
    #Save GP model
    gpmodel = GPModel(
        model=gp_model.cpu(),
        input_variables=input_variables,
        output_variables=output_variables,
        input_transformers=[input_transform],
        output_transformers=[output_transform],
    )

    model = gpmodel


with tempfile.TemporaryDirectory() as temp_dir:
    if model_type != 'GP':
        model.dump(file=os.path.join(temp_dir, experiment+'.yml'), save_jit=True )
    else:
        model.dump(file=os.path.join(temp_dir, experiment+'.yml'), save_models=True )
    # Upload the model to the database
    # - Load the files that were just created into a dictionary
    print("Loading model from temp dir")
    with open(os.path.join(temp_dir, experiment+'.yml')) as f:
        yaml_file_content = f.read()
    document = {
        'experiment': experiment,
        'model_type': model_type,
        'yaml_file_content': yaml_file_content
    }
    print(document)
    model_info = yaml.safe_load(yaml_file_content)
    for filename in [ model_info['model'] ] + model_info['input_transformers'] + model_info['output_transformers']:
        with open(os.path.join(temp_dir, filename), 'rb') as f:
            document[filename] = f.read()
    # - Check whether there is already a model in the database
    query = {'experiment': experiment, 'model_type': model_type}
    count = db['models'].count_documents(query)
    # - Upload/replace the model in the database
    if count > 1:
        print(f"Multiple models found for experiment: {experiment} and model type: {model_type}! Removing them.")
        db['models'].delete_many(query)
    elif count == 1:
        print(f"Model already exists for experiment: {experiment} and model type: {model_type}! Removing it.")
        db['models'].delete_one(query)
    print("Uploading new model to database")
    db['models'].insert_one(document)
    print("Model uploaded to database")
