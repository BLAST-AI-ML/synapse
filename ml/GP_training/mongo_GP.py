#!/usr/bin/env python
## This notebook includes simulation and experimental data
## in a database using PyMongo

import pandas as pd
import matplotlib.pyplot as plt
import torch
from botorch.models.transforms.input import AffineInputTransform
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
import pymongo
import pandas as pd
import os
import re
import yaml
from lume_model.models import TorchModel
from lume_model.variables import ScalarVariable, DistributionVariable
from lume_model.models.gp_model import GPModel
import sys
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

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
path_to_IFE_sf_src = "/global/cfs/cdirs/m558/superfacility/git"
path_to_IFE_ml = "/global/cfs/cdirs/m558/superfacility/git/ml/GP_training"
sys.path.append(path_to_IFE_ml)

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


# Train GP
if setup == 'ip2':
    gp_model = MultiTaskGP(
        torch.tensor( norm_df[['experiment_flag']+input_names].values ),
        torch.tensor( norm_df[output_names].values ),
        task_feature=0,
        covar_module=ScaleKernel(MaternKernel(nu=1.5)),
        outcome_transform=None,
    )
    cov = gp_model.task_covar_module._eval_covar_matrix()
    print( 'Correlation: ', cov[1,0]/torch.sqrt(cov[0,0]*cov[1,1]).item() )
elif setup in ['qed_ip2', 'acave']:
    gp_model = SingleTaskGP(
        torch.tensor(norm_df[input_names].values, dtype=torch.float64),
        torch.tensor(norm_df[output_names].values, dtype=torch.float64),
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

if setup == 'ip2':
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

model = GPModel(
    model=gp_model, 
    input_variables=input_variables_ordered,
    output_variables=output_variables,
    input_transformers=[input_transform],
    output_transformers=[output_transform],
)

dump( file=os.path.join(path_to_IFE_sf_src+'/ml/GP_training/saved_models', setup+'.yml'),     
save_models=True  
)