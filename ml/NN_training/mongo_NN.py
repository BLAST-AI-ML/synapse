#!/usr/bin/env python
## This notebook includes simulation and experimental data
## in a database using PyMongo
## Author : Revathi Jambunathan
## Date : January, 2025

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

# Select experimental setup for which we are training a model
setup = "qed_ip2"

# Open credential file for database
with open(os.path.join(os.getenv('HOME'), 'db.profile')) as f:
    db_profile = f.read()

# Connect to the MongoDB database with read-only access
db = pymongo.MongoClient(
    host="mongodb05.nersc.gov",
    username="bella_sf_ro",
    password=re.findall('SF_DB_READONLY_PASSWORD=(.+)', db_profile)[0],
    authSource="bella_sf")["bella_sf"]

# Extract data from the database as pandas dataframe
collection=db[setup]
df = pd.DataFrame( list(collection.find()) )

# Extract the name of inputs and outputs for this setup
#path_to_IFE_sf_src = "/global/homes/r/rjnathan/Codes/2024_IFE-superfacility/"
path_to_IFE_sf_src = "/global/cfs/cdirs/m558/superfacility/git"
path_to_IFE_ml = "/global/cfs/cdirs/m558/superfacility/git/ml/NN_training"
sys.path.append(path_to_IFE_ml)
from Neural_Net_Classes import CombinedNN as CombinedNN

with open(path_to_IFE_sf_src+"/dashboard/config/variables.yml") as f:
    yaml_dict = yaml.safe_load( f.read() )
input_variables = yaml_dict[setup]["input_variables"]
input_names = [ v['name'] for v in input_variables.values() ] 
output_variables = yaml_dict[setup]["output_variables"]
output_names = [ v['name'] for v in output_variables.values() ]

#Normalize with Affine Input Transformer
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
model.dump( file=os.path.join(path_to_IFE_sf_src+'/ml/NN_training/saved_models', setup+'.yml'), save_jit=True )

