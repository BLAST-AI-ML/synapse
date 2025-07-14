#!/usr/bin/env python
## This notebook includes simulation and experimental data
## in a database using PyMongo
## Author : Revathi Jambunathan
## Date : January, 2025
import time
import tempfile
import argparse
import torch
from botorch.models.transforms.input import AffineInputTransform
from botorch.models import MultiTaskGP, SingleTaskGP
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

# Extract the name of inputs and outputs for this experiment
path_to_IFE_sf_src = "/global/cfs/cdirs/m558/superfacility/"
path_to_IFE_ml = "/global/cfs/cdirs/m558/superfacility/model_training/src/"
sys.path.append(path_to_IFE_ml)
from Neural_Net_Classes import CombinedNN as CombinedNN

with open("/global/cfs/cdirs/m558/superfacility/model_training/src/variables.yml") as f:
    yaml_dict = yaml.safe_load( f.read() )
input_variables = yaml_dict[experiment]["input_variables"]
input_names = [ v['name'] for v in input_variables.values() ]
output_variables = yaml_dict[experiment]["output_variables"]
output_names = [ v['name'] for v in output_variables.values() ]

# Extract data from the database as pandas dataframe
collection = db[experiment]
df_exp = pd.DataFrame(db[experiment].find({"experiment_flag": 1}))
df_sim = pd.DataFrame(db[experiment].find({"experiment_flag": 0}))
# Apply calibration to the simulation results
simulation_calibration = yaml_dict[experiment]["simulation_calibration"]
for _, value in simulation_calibration.items():
    sim_name = value["name"]
    exp_name = value["depends_on"]
    df_sim[exp_name] = df_sim[sim_name] / value["alpha"] + value["beta"]

# Concatenate experimental and simulation data for training and validation
variables = input_names + output_names + ['experiment_flag']
if model_type != 'GP':
    #Split exp and sim data into training and validation data with 80:20 ratio, selected randomly
    exp_train_df, exp_val_df = train_test_split(df_exp, test_size=0.2, random_state=None, shuffle=True)# 20% of the data will go in validation test, no fixing the 
    sim_train_df, sim_val_df = train_test_split(df_sim, test_size=0.2, random_state=None, shuffle=True)#random_state will ensure the seed is different everytime, data will be shuffled randomly before splitting
    df_train = pd.concat( (exp_train_df[variables], sim_train_df[variables]) )
    df_val = pd.concat( (exp_val_df[variables], sim_val_df[variables]) )

else:
    # No split: all the data is training data
    df_train = pd.concate( (df_exp[variables], df_sim[variables]) )
    
# Normalize with Affine Input Transformer
# Define the input and output normalizations
X_train = torch.tensor( df_train[ input_names ].values, dtype=torch.float )
input_transform = AffineInputTransform(
    len(input_names),
    coefficient=X_train.std(axis=0),
    offset=X_train.mean(axis=0)
)
y_train = torch.tensor( df_train[ output_names ].values, dtype=torch.float )
output_transform = AffineInputTransform(
    len(output_names),
    coefficient=y_train.std(axis=0),
    offset=y_train.mean(axis=0)
)

# Apply normalization to the training data set
norm_df_train = df_train.copy()
norm_df_train[input_names] = input_transform( torch.tensor( df_train[input_names].values ) )
norm_df_train[output_names] = output_transform( torch.tensor( df_train[output_names].values ) )

norm_expt_inputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==1][input_names].values, dtype=torch.float)
norm_expt_outputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==1][output_names].values, dtype=torch.float)
norm_sim_inputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==0][input_names].values, dtype=torch.float)
norm_sim_outputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==0][output_names].values, dtype=torch.float)


######################################################
# Neural Net and Ensemble Creation and training
######################################################
if model_type != 'GP':
    # Saving the Lume Model - TO do for combined NN
    path_to_save = path_to_IFE_sf_src+'/ml/saved_models/NN_training/'
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


    NN_start_time = time.time()
    if model_type == 'NN':
        num_models = 1
    elif model_type == 'ensemble_NN':
        num_models = 10

    ensemble = []
    for i in range(num_models):
        model = CombinedNN(len(input_names), len(output_names), learning_rate=0.0001)
        model.train_model(
            norm_sim_inputs_train, norm_sim_outputs_train,
            norm_expt_inputs_train, norm_expt_outputs_train,
            norm_sim_inputs_val, norm_sim_outputs_val,
            norm_expt_inputs_val, norm_expt_outputs_val,
            num_epochs=20000)
        print(f'Model_{i+1} trained')
        ensemble.append(model)


    torch_models = []
    for model_nn in ensemble:
        calibration_transform = AffineInputTransform(
            len(output_names),
            coefficient=model_nn.sim_to_exp_calibration.weight.clone(),
            offset=model_nn.sim_to_exp_calibration.bias.clone() )

        # Fix mismatch in name between the config file and the expected lume-model format
        for k in input_variables:
            print(input_variables[k])
            input_variables[k]['default_value'] = input_variables[k]['default']

        model = TorchModel(
            model=model_nn,
            input_variables=[ ScalarVariable(**input_variables[k]) for k in input_variables.keys() ],
            output_variables=[ ScalarVariable(**output_variables[k]) for k in output_variables.keys() ],
            input_transformers=[input_transform],
            output_transformers=[calibration_transform,output_transform] # saving calibration before normalization
        )
        if num_models == 1:
            #Save single NN and break
            #model.dump( file=os.path.join(path_to_save, experiment+'.yml'), save_jit=True )
            #print(f"Model saved to {path_to_save}")
            end_time = time.time()
            break
        torch_models.append(model)

    #Save Ensemble
    if num_models > 1:
        model = NNEnsemble(
        models=torch_models,
        input_variables=[ ScalarVariable(**input_variables[k]) for k in input_variables.keys() ],
        output_variables=[ DistributionVariable(**output_variables[k]) for k in output_variables.keys() ]
        )
        #model.dump( file=os.path.join(path_to_save, experiment+'ensemble.yml'), save_jit=True )
        end_time = time.time()

    elapsed_time = end_time - start_time
    data_time = NN_start_time - start_time
    NN_time = end_time - NN_start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Data prep time taken: {data_time:.2f} seconds")
    print(f"NN time taken: {NN_time:.2f} seconds")

###############################################################
# Guassian Process Creation and training
###############################################################
else:
    if experiment != 'acave':
        gp_model = MultiTaskGP(
            torch.tensor( norm_df_train[['experiment_flag']+input_names].values ),
            torch.tensor( norm_df_train[output_names].values ),
            task_feature=0,
            covar_module=ScaleKernel(MaternKernel(nu=1.5)),
            outcome_transform=None,
        )
        cov = gp_model.task_covar_module._eval_covar_matrix()
        print( 'Correlation: ', cov[1,0]/torch.sqrt(cov[0,0]*cov[1,1]).item() )

    else:
        gp_model = SingleTaskGP(
            torch.tensor(norm_df_train[input_names].values, dtype=torch.float64),
            torch.tensor(norm_df_train[output_names].values, dtype=torch.float64),
            covar_module=ScaleKernel(MaternKernel(nu=1.5)),
            outcome_transform=None,
        )

    # Fit the model
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_mll(mll)

    # Fix mismatch in name between the config file and the expected lume-model format
    for k in input_variables:
        print(input_variables[k])
        input_variables[k]['default_value'] = input_variables[k]['default']
        del input_variables[k]['default']

    input_variables = [ ScalarVariable(**input_variables[k]) for k in input_variables.keys() ]

    if experiment != 'acave':
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
    model = GPModel(
        model=gp_model,
        input_variables=input_variables,
        output_variables=output_variables,
        input_transformers=[input_transform],
        output_transformers=[output_transform],
    )

    #path_to_save = path_to_IFE_sf_src+'/ml/saved_models/GP_training/'
    #model.dump( file=os.path.join(path_to_save, experiment+'.yml'), save_models=True )
    print(f"Model saved to {temp}")


with tempfile.TemporaryDirectory() as temp_dir:
    if model_type != 'GP':
        model.dump(file=os.path.join(temp_dir, experiment+'.yml'), save_jit=True )
    else:
        model.dump(file=os.path.join(temp_dir, experiment+'.yml'), save_models=True )
    # Upload the model to the database
    # - Load the files that were just created into a dictionary
    print(f"Loading model from {temp_dir}")
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
        count = db['models'].count_documents(query)
    if count == 0:
        print("Uploading new model to database")
        db['models'].insert_one(document)
        print("Model uploaded to database")
    elif count == 1:
        print('Model already exists in database ; updating it.')
        db['models'].update_one(query, {'$set': document})
    else:
        # Raise error, this should not happen
        raise ValueError(f"Multiple models found for experiment: {experiment} and model type: {model_type}!")
    
    print("Model updated in database")
