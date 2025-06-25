#To run this
#python3 mongo_ensemble_NN.py <setup name> <num_models>
import pandas as pd
import torch
from botorch.models.transforms.input import AffineInputTransform
import pymongo
import os
import re
import yaml
from lume_model.models import TorchModel
from lume_model.variables import ScalarVariable
import sys
import numpy as np

# Select experimental setup for which we are training a model
setup = sys.argv[1]
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
path_to_IFE_sf_src = "/global/cfs/cdirs/m558/superfacility/"
path_to_IFE_ml = "/global/cfs/cdirs/m558/superfacility/model_training/src/"
sys.path.append(path_to_IFE_ml)
from Neural_Net_Classes import CombinedNN as CombinedNN

with open("/global/cfs/cdirs/m558/superfacility/model_training/src/variables.yml") as f:
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

#Generate and Train Ensemble
num_models = int(sys.argv[2])
ensemble = []
for i in range(num_models):
    model = CombinedNN(len(input_names), len(output_names), learning_rate=0.0001)
    model.train_model(norm_sim_inputs_training, norm_sim_outputs_training,
                     norm_expt_inputs_training, norm_expt_outputs_training, num_epochs=10000)
    print(f"\nModel_{i+1} trained\n")
    ensemble.append(model)

#Find Alphas and Betas (include mean and std)
models = [f"Model {i+1}" for i in range(len(ensemble))]
alphas = []
betas = []

for model in ensemble:
    alpha = model.sim_to_exp_calibration.weight.data.item()
    beta = model.sim_to_exp_calibration.bias.data.item()   
    alphas.append(alpha)
    betas.append(beta)
    
mean_alpha = np.mean(np.array(alphas))
std_alpha = np.std(np.array(alphas))
mean_beta = np.mean(np.array(betas))
std_beta = np.std(np.array(betas))

alphas_and_betas = pd.DataFrame({
    'model': models,
    'alpha': alphas,
    'beta': betas
})
print(alphas_and_betas)

print(f'\nAlpha Mean: {mean_alpha:.4f}\nAlpha Std: {std_alpha:.4f}\n\nBeta Mean: {mean_beta:.4f}\nBeta Std: {std_beta:.4f}')

#Save Ensemble
path = f'/global/homes/e/erod/2024_IFE-superfacility/ml/NN_training/Ensemble_Models/{setup}/'
os.makedirs(path, exist_ok=True)
# Save alphas, betas, and their statistics to a text file
summary_path = os.path.join(path, "alpha_beta_summary.txt")
with open(summary_path, "w") as f:
    f.write("Alpha and Beta Calibration Parameters per Model:\n")
    f.write(alphas_and_betas.to_string(index=False))
    f.write("\n\n")
    f.write(f"Alpha Mean: {mean_alpha:.4f}\n")
    f.write(f"Alpha Std: {std_alpha:.4f}\n")
    f.write(f"Beta Mean: {mean_beta:.4f}\n")
    f.write(f"Beta Std: {std_beta:.4f}\n")

torch_models = []

for i, model_nn in enumerate(ensemble):
    calibration_transform = AffineInputTransform(
            len(output_names), 
            coefficient=model_nn.sim_to_exp_calibration.weight.clone(), 
            offset=model_nn.sim_to_exp_calibration.bias.clone()
        )
    
    for k in input_variables:
        #print(input_variables[k])
        input_variables[k]['default_value'] = input_variables[k]['default']
        
    torch_model = TorchModel(
        model=model_nn,
        input_variables=[ ScalarVariable(**input_variables[k]) for k in input_variables.keys() ],
        output_variables=[ ScalarVariable(**output_variables[k]) for k in output_variables.keys() ],
        input_transformers=[input_transform],
        output_transformers=[calibration_transform,output_transform] # saving calibration before normalization
        )
    torch_model.dump( file=os.path.join(path, setup+'_model'+str(i)+'.yml'), save_jit=True )
