#!/usr/bin/env python3

import os
import re
import numpy as np
import pymongo
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import c, mu_0
from datetime import datetime

try:
    from mip4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

def analyze_simulation():

    # Get current directory
    data_directory = os.path.join( os.getcwd(), 'diags' )
    ts = OpenPMDTimeSeries( os.path.join(data_directory, 'out') )

    # Additional metadata
    data = {}
    data['experiment_flag'] = 0
    data['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['data_directory'] = data_directory

    # Parse the warpx_used_output
    with open('warpx_used_inputs') as f:
        text = f.read()
        data['plasma_gradient_length'] = float( re.findall('my_constants\.plasma_gradient_length\s+= (.+)', text)[0] )
        data['target_to_focus_distance'] = float( re.findall('my_constants\.target_to_focus_distance\s+= (.+)', text)[0] )

    # Compute energy in the harmonics, above harmonic `min_harmonic`
    min_harmonic = 9
    wvl = 800e-9
    w0 = 2.7e-6
    By, info = ts.get_field(iteration=ts.iterations[-1], field="B", coord="y")
    dims = np.shape(By)
    dx, dz = info.dx, info.dz
    kx = 2*np.pi*np.fft.fftfreq(dims[0], d=info.dx)
    kz = 2*np.pi*np.fft.fftfreq(dims[1], d=info.dz)
    By_fft = np.fft.fftn(By)
    KX, KZ = np.meshgrid(kx, kz, indexing="ij")
    OM = c*np.sqrt(KX**2 + KZ**2)
    om0 = 2*np.pi*c/wvl*min_harmonic
    By_fft_cut = np.where(OM > om0, By_fft, 0)
    By_cut = np.fft.ifftn(By_fft_cut).real
    E = np.sum(By_cut**2*(1/mu_0))*dx*dz*2*w0
    data['energy_in_harmonics'] = E

    # Write to the data base
    db = pymongo.MongoClient(
        host="mongodb05.nersc.gov",
        username="bella_sf_admin",
        password=os.getenv("SF_DB_ADMIN_PASSWORD"),
        authSource="bella_sf")["bella_sf"]
    collection = db["qed_ip2"]
    collection.insert_one(data)

    # Create plots of the interation

if __name__ == '__main__':

    # When this script is called in optimas
    # it might be run by multiple processes
    # but we only need to run it once
    if rank == 0:
        analyze_simulation()
