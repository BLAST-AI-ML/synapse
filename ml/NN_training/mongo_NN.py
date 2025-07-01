#!/usr/bin/env python
## This notebook includes simulation and experimental data
## in a database using PyMongo
## Author : Revathi Jambunathan
## Date : January, 2025

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
from botorch.models.transforms.input import AffineInputTransform
import pymongo
import pandas as pd
import os
import re
import yaml
from lume_model.models import TorchModel
from lume_model.variables import ScalarVariable
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
    username="bella_sf_ro",
    password=re.findall('SF_DB_READONLY_PASSWORD=(.+)', db_profile)[0],
    authSource="bella_sf")["bella_sf"]

# Extract the name of inputs and outputs for this experiment
path_to_IFE_sf_src = "/global/cfs/cdirs/m558/superfacility/"
path_to_IFE_ml = "/global/cfs/cdirs/m558/superfacility/model_training/src/"
sys.path.append(path_to_IFE_ml)
from Neural_Net_Classes import CombinedNN as CombinedNN

with open("../../dashboard/config/variables.yml") as f:
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
# Concatenate experimental and simulation data
variables = input_names + output_names + ['experiment_flag']
df = pd.concat( (df_exp[variables], df_sim[variables]) )

# Normalize with Affine Input Transformer
# Define the input and output normalizations
X = torch.tensor( df[ input_names ].values, dtype=torch.float )
input_transform = AffineInputTransform( 
    len(input_names), 
    coefficient=X.std(axis=0), 
    offset=X.mean(axis=0)
)
y = torch.tensor( df[ output_names ].values, dtype=torch.float )
output_transform = AffineInputTransform( 
    len(output_names), 
    coefficient=y.std(axis=0),
    offset=y.mean(axis=0)
)

# Apply normalization to the data set
norm_df = df.copy()
norm_df[input_names] = input_transform( torch.tensor( df[input_names].values ) )
norm_df[output_names] = output_transform( torch.tensor( df[output_names].values ) )

norm_expt_inputs_training = torch.tensor( norm_df[norm_df.experiment_flag==1][input_names].values, dtype=torch.float)
norm_expt_outputs_training = torch.tensor( norm_df[norm_df.experiment_flag==1][output_names].values, dtype=torch.float)
norm_sim_inputs_training = torch.tensor( norm_df[norm_df.experiment_flag==0][input_names].values, dtype=torch.float)
norm_sim_outputs_training = torch.tensor( norm_df[norm_df.experiment_flag==0][output_names].values, dtype=torch.float)


# Train combined NN
calibrated_nn = CombinedNN( len(input_names), len(output_names), learning_rate=0.0005)
calibrated_nn.train_model(
    norm_sim_inputs_training, norm_sim_outputs_training,
    norm_expt_inputs_training, norm_expt_outputs_training, 
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
#model.dump( file=os.path.join(path_to_IFE_sf_src+'/ml/NN_training/saved_models', experiment+'.yml'), save_jit=True )
