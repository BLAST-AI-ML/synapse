#!/usr/bin/env python3

import os
import json
import pymongo
from openpmd_viewer.addons import LpaDiagnostics

# Get current directory
data_directory = os.path.join( os.getcwd(), 'diags/diag' )
ts = LpaDiagnostics( data_directory )

# Compute average wavelength at the last iteration
last_iteration = 3500
S, info = ts.get_laser_spectral_intensity(iteration=last_iteration, pol='x')
lambda_avg = np.average( 2*np.pi/info.k[1:], weight=S[1:] )

# Log results to the database

# Load input parameters
with open('input_params.json') as file:
    data = json.load(file)

# Add the mean wavelength

# Add date and folder

# Write to the data base
db = pymongo.MongoClient(
    host="mongodb05.nersc.gov",
    username="bella_sf_admin",
    password=os.getenv("SF_DB_ADMIN_PASSWORD"),
    authSource="bella_sf")["bella_sf"]
collection = db["acave"]
