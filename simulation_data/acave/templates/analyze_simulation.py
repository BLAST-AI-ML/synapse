#!/usr/bin/env python3

import os
import json
import numpy as np
import pymongo
import matplotlib.pyplot as plt
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c
from datetime import datetime

# Get current directory
data_directory = os.path.join( os.getcwd(), 'diags/diag' )
ts = LpaDiagnostics( data_directory )

# Load input parameters
with open('input_params.json') as file:
    data = json.load(file)
# Additional metadata
data['experiment_flag'] = 0
data['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data['data_directory'] = data_directory

# Compute average wavelength at the last iteration and add it
last_iteration = 3500
S, info = ts.get_laser_spectral_intensity(iteration=last_iteration, pol='x')
lambda_avg = np.average( 2*np.pi/info.k[1:], weights=S[1:] )
data['kHz_thorlab_spectrometer mean_wavelength'] = lambda_avg

# Write to the data base
db = pymongo.MongoClient(
    host="mongodb05.nersc.gov",
    username="bella_sf_admin",
    password=os.getenv("SF_DB_ADMIN_PASSWORD"),
    authSource="bella_sf")["bella_sf"]
collection = db["acave"]
collection.insert_one(data)

# Create plots of the interation
def visualize_iteration(iteration):
    plt.clf()

    plt.subplot(121)
    # Plot of the laser + ionized electron density
    rho, info = ts.get_field('rho_electrons', iteration=iteration)
    plt.imshow(abs(rho.T), cmap='Blues',
               extent=[1e6*info.zmin, 1e6*info.zmax, 1e6*info.rmin, 1e6*info.rmax],
               aspect='auto')
    Ex, info = ts.get_field('E', 'x', iteration=iteration)
    plt.imshow(abs(Ex.T), cmap='gist_heat_r',
               extent=[1e6*info.zmin, 1e6*info.zmax, 1e6*info.rmin, 1e6*info.rmax],
               aspect='auto', alpha=0.5)
    plt.xlabel('$z [\mu m]$')
    plt.ylabel('$x [\mu m]$')

    plt.subplot(122)
    # Plot of the laser spectrum
    S, info = ts.get_spectrum(iteration=iteration, pol='x')
    lambd = 2*np.pi*c/info.omega[1:]
    plt.xlabel(r'Wavelength[$\mu m$]')
    plt.plot( 1.e6*lambd, S[1:] )
    plt.xlim(0,3)

    plt.grid()
    plt.savefig('diags/plots/iteration_%05d.png' % iteration)

plt.figure(figsize=(8, 4))
ts.iterate( visualize_iteration )
