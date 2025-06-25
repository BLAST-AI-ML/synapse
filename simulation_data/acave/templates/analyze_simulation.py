#!/usr/bin/env python3

import os
import re
import json
import numpy as np
import pymongo
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from openpmd_viewer.addons import LpaDiagnostics
from scipy.constants import c
from datetime import datetime
from PIL import Image
import imageio.v2 as imageio

try:
    from mip4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

def analyze_simulation():

    # Get current directory
    data_directory = os.path.join( os.getcwd(), 'diags' )
    ts = LpaDiagnostics( os.path.join(data_directory, 'diag') )

    # Load input parameters
    with open('input_params.json') as file:
        data = json.load(file)
    # Additional metadata
    data['experiment_flag'] = 0
    data['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['data_directory'] = data_directory

    # Parse the warpx_used_output
    with open('warpx_used_inputs') as f:
        text = f.read()
        last_step = int( re.findall('max_step = (\d+)', text)[0] )
        dens_width = float( re.findall('my_constants\.dens_width = (.+)', text)[0] )
        n0 = float( re.findall('my_constants\.n0 = (.+)', text)[0] )
        density_function = re.findall('atoms\.density_function\(x,y,z\) = (.+)', text)[0]

    # Compute average wavelength at the last iteration and add it to the data
    last_iteration = last_step
    S, info = ts.get_laser_spectral_intensity(iteration=last_iteration, pol='x')
    lambda_avg = np.average( 2*np.pi/info.k[1:], weights=S[1:] )
    data['kHz_ThorlabsSpec MeanWavelength'] = lambda_avg*1e9 # convert from m to nm

    # Write to the data base
    db = pymongo.MongoClient(
        host="mongodb05.nersc.gov",
        username="bella_sf_admin",
        password=os.getenv("SF_DB_ADMIN_PASSWORD"),
        authSource="bella_sf")["bella_sf"]
    collection = db["acave"]
    collection.insert_one(data)

    # Create plots of the interation

    # First compute quantities across the interaction:
    # - Density
    z = np.linspace(0, 400e-6, 1000)
    exp = np.exp
    n = eval( density_function )
    # - a0
    a0 = ts.iterate( ts.get_a0, pol='x')
    # - Mean laser position
    def get_mean_laser_position(iteration):
        env, info = ts.get_laser_envelope( pol='x', iteration=iteration, slice_across='r' )
        return np.average( info.z, weights=env )
    z_laser = ts.iterate( get_mean_laser_position )

    def visualize_iteration(iteration):

        # Prepare figure
        plt.clf()
        fig = plt.gcf()
        gs = GridSpec(2, 2)

        fig.add_subplot(gs[0,:])
        # Plot of a0, density as a function of z
        plt.plot( 1e6*z, 1e-6*n )
        plt.xlabel('z[$\mu m]$')
        plt.ylabel('Atomic density [$cm^{-3}$]', color='b')
        # find laser position at the current iteration
        z0 = z_laser[ np.argmin(abs(ts.iterations-iteration)) ]
        plt.axvline(x=1e6*z_laser[ np.argmin(abs(ts.iterations-iteration)) ],
                color='k', ls='--')
        plt.twinx()
        plt.plot( 1e6*z_laser, a0, 'r' )
        plt.ylabel('a0', color='r')
        plt.ylim(0,1)

        fig.add_subplot(gs[1,0])
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

        fig.add_subplot(gs[1,1])
        # Plot of the laser spectrum
        S, info = ts.get_spectrum(iteration=iteration, pol='x')
        lambd = 2*np.pi*c/info.omega[1:]
        plt.xlabel(r'Wavelength[$\mu m$]')
        plt.plot( 1.e6*lambd, 1e3*S[1:] )
        plt.xlim(0,3)
        plt.ylim(0,10)

        plt.grid()
        plt.savefig('diags/plots/iteration_%05d.png' % iteration)

    plt.figure(figsize=(8, 8))
    ts.iterate( visualize_iteration )

    # Load images and convert to MP4
    image_files = ['diags/plots/iteration_%05d.png' % iteration for iteration in ts.iterations]
    with imageio.get_writer('diags/plots/animation.mp4', fps=5) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("animation.mp4 created successfully!")


if __name__ == '__main__':

    # When this script is called in optimas
    # it might be run by multiple processes
    # but we only need to run it once
    if rank == 0:
        analyze_simulation()
