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
import imageio.v2 as imageio
from transparent_imshow import transp_imshow

try:
    from mip4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

# General parameters of the analysis
pol = 'y'
uz_threshold = 10

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
        stage_length = float( re.findall(r'my_constants\.stage_length = (.+)', text)[0] )
        ramp_length = float( re.findall(r'my_constants\.ramp_length = (.+)', text)[0] )
        dopant_length = float( re.findall(r'my_constants\.dopant_length = (.+)', text)[0] )
        dopant_fraction = float( re.findall(r'my_constants\.dopant_fraction = (.+)', text)[0] )
        n_upstream_atom = float( re.findall(r'my_constants\.n_upstream_atom = (.+)', text)[0] )
        n_downstream_atom = float( re.findall(r'my_constants\.n_downstream_atom = (.+)', text)[0] )
        hydrogen_density_function = re.findall(r'hydrogen1\.density_function\(x,y,z\) = (.+)', text)[0]
        nitrogen_density_function = re.findall(r'nitrogen1\.density_function\(x,y,z\) = (.+)', text)[0]

    # Compute red/blue shift: wavelength such that 13.5%/86.5% of the spectrum energy is below
    S, info = ts.get_laser_spectral_intensity(
        iteration=ts.iterations[-1], pol=pol)
    cumulated_S = np.cumsum(S)/np.sum(S)
    kr = np.argmin(abs(cumulated_S-0.135))
    kb = np.argmin(abs(cumulated_S-0.865))
    lambda_r = 2*np.pi/info.k[kr]
    lambda_b = 2*np.pi/info.k[kb]
    # Add to the data base
    data['SPEC-AA-Hamamastsu lambda_r'] = lambda_r*1e9 # convert from m to nm
    data['SPEC-AA-Hamamastsu lambda_b'] = lambda_b*1e9 # convert from m to nm

    # Create plots of the interaction

    # First compute quantities across the interaction:
    # - Density
    z = np.linspace(0, stage_length, 1000)
    sqrt = np.sqrt
    n_H = eval( hydrogen_density_function )
    n_N = eval( nitrogen_density_function )
    # - a0
    a0 = ts.iterate( ts.get_a0, pol=pol)
    w0 = ts.iterate( ts.get_laser_waist, pol=pol)
    # - Mean laser position
    def get_mean_laser_position(iteration):
        env, info = ts.get_laser_envelope( pol=pol, iteration=iteration, slice_across='r' )
        return np.average( info.z, weights=env )
    z_laser = ts.iterate( get_mean_laser_position )
    # - beam charge
    Q = ts.iterate( ts.get_charge, species='electrons_n1',
                   select={'uz':[uz_threshold, None]})
    no_trapped_electrons = np.all(Q == 0) # check if there are any trapped electrons in this simulation
    # - energy and energy spread
    gamma, dgamma = ts.iterate( ts.get_mean_gamma, species='electrons_n1')

    data['Beam mean energy [GeV]'] = gamma[-1]*0.511e-3 # convert from m to nm
    data['Beam energy spread [%]'] = 100*dgamma[-1]/gamma[-1]
    data['Trapped charge [pC]'] = -Q[-1]*1e12

    # Extract divergence at the last iteration
    div_x, _ = ts.get_divergence(iteration=ts.iterations[-1], species='electrons_n1')
    data['Beam RMS div x [mrad]'] = div_x*1e3 # convert from rad to mrad

    # Write to the data base
    db = pymongo.MongoClient(
        host="mongodb05.nersc.gov",
        username="bella_sf_admin",
        password=os.getenv("SF_DB_ADMIN_PASSWORD"),
        authSource="bella_sf")["bella_sf"]
    collection = db["staging_injector"]
    collection.insert_one(data)

    def visualize_iteration(iteration):

        # Prepare figure
        plt.clf()
        fig = plt.gcf()
        gs = GridSpec(4, 2)

        # find laser position at the current iteration
        current_z_las = z_laser[ np.argmin(abs(ts.iterations-iteration)) ]
        ct = c*ts.t[abs(iteration-ts.iterations).argmin()]

         # Plot of density as a function of z
        fig.add_subplot(gs[0,0])
        plt.plot( 1e2*z, 1e-6*n_H, label='Hydrogen' )
        plt.plot( 1e2*z, 1e-6*n_N, label='Nitrogen' )
        plt.ylabel('Atomic density [$cm^{-3}$]')
        plt.legend(loc=0)
        plt.grid()
        plt.axvline(x=1e2*current_z_las, color='k', ls='--')
        plt.xlim(0, 1e2*stage_length)

        # Plot of charge
        fig.add_subplot(gs[1,0])
        plt.plot( 1e2*z_laser, -1e12*Q, color='b' )
        plt.ylabel('Trapped charge [pC]')
        plt.grid()
        plt.axvline(x=1e2*current_z_las, color='k', ls='--')
        plt.xlim(0, 1e2*stage_length)

        # Plot of energy and energy spread
        fig.add_subplot(gs[2,0])
        # Skip this plot if there are no electrons
        if not no_trapped_electrons:
            plt.plot( 1e2*z_laser, 0.511e-3*gamma, color='b' )
            plt.fill_between( 1e2*z_laser, 0.511e-3*(gamma-dgamma),
                            0.511e-3*(gamma+dgamma), color='b', alpha=0.3)
            plt.ylabel('Beam energy [GeV]')
            plt.grid()
            plt.axvline(x=1e2*current_z_las, color='k', ls='--')
            plt.xlim(0, 1e2*stage_length)

        # Plot of a0 and w0 as a function of z
        fig.add_subplot(gs[3,0])
        plt.plot( 1e2*z_laser, a0, 'r' )
        plt.ylabel('a0', color='r')
        plt.grid()
        plt.xlabel(r'z[cm]')
        plt.twinx()
        plt.plot( 1e2*z_laser, 1e6*w0, 'purple' )
        plt.ylabel(r'waist [$\mu$m]', color='purple')
        plt.axvline(x=1e2*current_z_las, color='k', ls='--')
        plt.xlim(0, 1e2*stage_length)

        # Plot of the laser + accelerating field
        fig.add_subplot(gs[0:2,1])
        Ez, info = ts.get_field('E', 'z', iteration=iteration)
        plt.imshow( abs(Ez.T), cmap='Blues', interpolation='bicubic',
                extent=[1e6*(info.zmin-ct), 1e6*(info.zmax-ct),
                        1e6*info.rmin, 1e6*info.rmax],
                aspect='auto', vmin=0, vmax=5e10)
        plt.ylim(-100, 100)
        env, info = ts.get_laser_envelope(iteration=iteration, pol=pol)
        transp_imshow(abs(env.T), cmap='gist_heat_r',
            aspect='auto', gam=0.3,
            extent=[1e6*(info.zmin-ct), 1e6*(info.zmax-ct),
                    1e6*info.rmin, 1e6*info.rmax] )
        xp, zp = ts.get_particle(['x', 'z'], iteration=iteration,
            species='electrons_n1')
        plt.plot(1e6*(zp-ct), 1e6*xp, '.', color='orange', ms=1 )
        plt.xlabel(r'$z - ct \;[\mu m]$')
        plt.ylabel(r'$x [\mu m]$')
        plt.title(r'$|E_z|$ (blue), laser envelope (red)')
        plt.xlim(-95, 0)

        # Plot of the electron energy spectrum
        fig.add_subplot(gs[2,1])
        # Skip this plot if there are no electrons
        if not no_trapped_electrons:
            uz, w = ts.get_particle(['uz', 'w'], iteration=iteration,
                select={'uz':[uz_threshold, None]}, species='electrons_n1')
            plt.hist( 0.511e-3*uz, weights=w, bins=200,
                    range=[0, 1.2*0.511e-3*np.nanmax(gamma)] )
            plt.xlabel(r'Energy [GeV]')
            plt.title('Electron energy spectrum')

        # Plot of the laser spectrum
        fig.add_subplot(gs[3,1])
        S, info = ts.get_spectrum(iteration=iteration, pol=pol)
        lambd = 2*np.pi*c/info.omega[1:]
        plt.xlabel(r'Wavelength [$\mu m$]')
        plt.title('Laser spectrum')
        plt.plot( 1.e6*lambd, 1e3*S[1:], color='r' )
        plt.xlim(0.5,1.2)

        plt.subplots_adjust(hspace=0.5)
        plt.savefig('diags/plots/iteration_%05d.png' % iteration)

    # Create directory
    os.makedirs( 'diags/plots/', exist_ok=True )

    # Create images
    plt.figure(figsize=(12, 12))
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
