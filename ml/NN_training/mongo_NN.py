#!/usr/bin/env python
## This notebook includes simulation and experimental data
## in a database using PyMongo
## Author : Revathi Jambunathan
## Date : January, 2025

import argparse
import torch
from botorch.models.transforms.input import AffineInputTransform
import pymongo
import pandas as pd
import os
import re
import yaml
from lume_model.models import TorchModel
from lume_model.variables import ScalarVariable
from sklearn.model_selection import train_test_split
import sys


# define parser
parser = argparse.ArgumentParser()
# add argument
parser.add_argument(
    "--experiment",
    help="name/tag of the experiment",
    type=str,
    required=True,
)
# parse arguments
args = parser.parse_args()

# Select experiment for which we are training a model
experiment = args.experiment

# Open credential file for database
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

#Split exp and sim data into training and validation data with 80:20 ratio, selected randomly
exp_train_df, exp_val_df = train_test_split(df_exp, test_size=0.2, random_state=None, shuffle=True)# 20% of the data will go in validation test, no fixing the random_state will ensure the seed is different everytime, data will be shuffled randomly before splitting
sim_train_df, sim_val_df = train_test_split(df_sim, test_size=0.2, random_state=None, shuffle=True)

# Concatenate experimental and simulation data for training and validation
variables = input_names + output_names + ['experiment_flag']
df_train = pd.concat( (exp_train_df[variables], sim_train_df[variables]) )
df_val = pd.concat( (exp_val_df[variables], sim_val_df[variables]) )

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

X_val = torch.tensor( df_val[ input_names ].values, dtype=torch.float )
input_transform = AffineInputTransform(
    len(input_names),
    coefficient=X_val.std(axis=0),
    offset=X_val.mean(axis=0)
)
y_val = torch.tensor( df_val[ output_names ].values, dtype=torch.float )
output_transform = AffineInputTransform(
    len(output_names),
    coefficient=y_val.std(axis=0),
    offset=y_val.mean(axis=0)
)

# Apply normalization to the training data set
norm_df_train = df_train.copy()
norm_df_train[input_names] = input_transform( torch.tensor( df_train[input_names].values ) )
norm_df_train[output_names] = output_transform( torch.tensor( df_train[output_names].values ) )

norm_expt_inputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==1][input_names].values, dtype=torch.float)
norm_expt_outputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==1][output_names].values, dtype=torch.float)
norm_sim_inputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==0][input_names].values, dtype=torch.float)
norm_sim_outputs_train = torch.tensor( norm_df_train[norm_df_train.experiment_flag==0][output_names].values, dtype=torch.float)

# Apply normalization to the validation data set
norm_df_val = df_val.copy()
norm_df_val[input_names] = input_transform( torch.tensor( df_val[input_names].values ) )
norm_df_val[output_names] = output_transform( torch.tensor( df_val[output_names].values ) )

norm_expt_inputs_val = torch.tensor( norm_df_val[norm_df_val.experiment_flag==1][input_names].values, dtype=torch.float)
norm_expt_outputs_val = torch.tensor( norm_df_val[norm_df_val.experiment_flag==1][output_names].values, dtype=torch.float)
norm_sim_inputs_val = torch.tensor( norm_df_val[norm_df_val.experiment_flag==0][input_names].values, dtype=torch.float)
norm_sim_outputs_val = torch.tensor( norm_df_val[norm_df_val.experiment_flag==0][output_names].values, dtype=torch.float)


# Train combined NN
calibrated_nn = CombinedNN( len(input_names), len(output_names), learning_rate=0.0005)
calibrated_nn.train_model(
    norm_sim_inputs_train, norm_sim_outputs_train,
    norm_expt_inputs_train, norm_expt_outputs_train,
    norm_sim_inputs_val, norm_sim_outputs_val,
    norm_expt_inputs_val, norm_expt_outputs_val,
    num_epochs=20000)


# Saving the Lume Model - TO do for combined NN
calibration_transform = AffineInputTransform(
    len(output_names),
    coefficient=calibrated_nn.sim_to_exp_calibration.weight.clone(),
    offset=calibrated_nn.sim_to_exp_calibration.bias.clone() )

# Fix mismatch in name between the config file and the expected lume-model format
for k in input_variables:
    print(input_variables[k])
    input_variables[k]['default_value'] = input_variables[k]['default']
    del input_variables[k]['default']

model = TorchModel(
    model=calibrated_nn,
    input_variables=[ ScalarVariable(**input_variables[k]) for k in input_variables.keys() ],
    output_variables=[ ScalarVariable(**output_variables[k]) for k in output_variables.keys() ],
    input_transformers=[input_transform],
    output_transformers=[calibration_transform,output_transform] # saving calibration before normalization
)

path_to_save = path_to_IFE_sf_src+'/ml/saved_models/NN_training/'
model.dump( file=os.path.join(path_to_save, experiment+'.yml'), save_jit=True )
print(f"Model saved to {path_to_save}")

# Upload the model to the database
model_choice = 'NN'
# - Load the files that were just created into a dictionary
print(f"Loading model from {path_to_save}")
with open(os.path.join(path_to_save, experiment+'.yml')) as f:
    yaml_file_content = f.read()
document = {
    'experiment': experiment,
    'model_type': model_choice,
    'yaml_file_content': yaml_file_content
}
print(document)
model_info = yaml.safe_load(yaml_file_content)
for filename in [ model_info['model'] ] + model_info['input_transformers'] + model_info['output_transformers']:
    with open(os.path.join(path_to_save, filename), 'rb') as f:
        document[filename] = f.read()
# - Check whether there is already a model in the database
query = {'experiment': experiment, 'model_type': model_choice}
count = db['models'].count_documents(query)
# - Upload/replace the model in the database
if count > 1:
    print(f"Multiple models found for experiment: {experiment} and model type: {model_choice}! Removing them.")
    db['models'].delete_many(query)
if count == 0 or count > 1:
    print("Uploading new model to database")
    db['models'].insert_one(document)
    print("Model uploaded to database")
elif count == 1:
    print('Model already exists in database ; updating it.')
    db['models'].update_one(query, {'$set': document})
print("Model updated in database")