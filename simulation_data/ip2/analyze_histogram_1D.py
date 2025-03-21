#!/usr/bin/env python3

# This script analyzes our 1D energy histograms for the IFE Superfacilities LDRD.
# We want a single number as a result: the total number of particles within our energy range of interest.
# The bins are expected to be in normalized momentum units.
#
# This file also contains a script to produce a waterfall plot of the energy histograms over time.
# Its purpose is to produce images that let the user get a quick idea of what the dynamics of the acceleration look like.
# We produce histograms for:
# - all forward flying protons behind the cold target rear surface
# - forward flying protons in an opening angle according to RCF stack detectors
# - forward flying protons in an opening angle according to a pinhole for a Thomson parabola detector
#

import argparse

# regex for formatting
import re
import os
import numpy as np
import pandas as pd
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def analyze_hist1D(
        filepath='./diags/reducedfiles/histuH.txt',
        Ekin_MeV_lo=5,
        Ekin_MeV_hi=20,
        time_readout_fs=600
):

    df = pd.read_csv(filepath,delimiter=r'\s+')
    # the columns look like this:
    #     #[0]step() [1]time(s) [2]bin1=0.000220() [3]bin2=0.000660() [4]bin3=0.001100()

    # matches words, strings surrounded by " ' ", dots, minus signs and e for scientific notation in numbers
    nested_list = [re.findall(r"[\w'\.]+",col) for col in df.columns]

    index = pd.MultiIndex.from_tuples(nested_list, names=('column#', 'name','bin value'))

    df.columns=(index)

    steps = df.values[:,0].astype(int)
    time = df.values[:,1]
    data = df.values[:,2:]
    edge_vals = np.array([float(row[2]) for row in df.columns[2:]])

    # proton rest energy in eV
    mpc2 = sc.m_p/sc.electron_volt * sc.c**2

    fs = 1.e-15
    MeV = 1.e6
    time_fs = time / fs

    edges_MeV = (np.sqrt(edge_vals**2 + 1)-1) * mpc2 / MeV
    my_filter = (edges_MeV >= Ekin_MeV_lo)*(edges_MeV < Ekin_MeV_hi)

    time_readout = time_readout_fs * fs
    time_difference = (df[('1','time','s')] - time_readout).abs()

    # Find the row with the minimum difference
    closest_entry_idx = time_difference.idxmin()
    closest_entry = df.loc[closest_entry_idx]


    bins_data = closest_entry.iloc[2:].values

    num_in_interval = np.sum(bins_data[my_filter])

    return num_in_interval

def plot_histogram_waterfall(
        sim_dir='./exploration/evaluations/sim0000/',
        detector_str="fw",
        ymin=0,
        ymax=120
):
    """
    Plots a waterfall histogram of kinetic energy vs. time for a given simulation directory and detector.

    Parameters:
    sim_dir (str, optional): Path to the simulation directory. Defaults to './exploration/evaluations/sim0000/'.
    detector_str (str, optional): Detector string (e.g. "fw" for forward detector). Defaults to "fw". (Other options: "tp", "rcf")
    ymin (float, optional): Minimum y-axis value (kinetic energy in MeV). Defaults to 0.
    ymax (float, optional): Maximum y-axis value (kinetic energy in MeV). Defaults to 120.

    Returns:
    None

    The function reads a histogram file from the simulation directory, processes the data, and creates a waterfall plot
    of kinetic energy vs. time. The plot is saved to a file in the simulation directory's "analysis/waterfall/{detector_str}" subdirectory.
    """

    filepath = f'./{sim_dir}/diags/reducedfiles/histuH_{detector_str}.txt'

    df = pd.read_csv(filepath,delimiter=r'\s+')
    # the columns look like this:
    #     #[0]step() [1]time(s) [2]bin1=0.000220() [3]bin2=0.000660() [4]bin3=0.001100()

    # matches words, strings surrounded by " ' ", dots, minus signs and e for scientific notation in numbers
    nested_list = [re.findall(r"[\w'\.]+",col) for col in df.columns]

    index = pd.MultiIndex.from_tuples(nested_list, names=('column#', 'name','bin value'))

    df.columns=(index)

    steps = df.values[:,0].astype(int)
    time = df.values[:,1]
    data = df.values[:,2:]
    edge_vals = np.array([float(row[2]) for row in df.columns[2:]])

    # proton rest energy in eV
    mpc2 = sc.m_p/sc.electron_volt * sc.c**2

    fs = 1.e-15
    MeV = 1.e6
    time_fs = time / fs
    edges_MeV = (np.sqrt(edge_vals**2 + 1)-1) * mpc2 / MeV

    extent = np.array([
        time_fs[0], time_fs[-1],
        edges_MeV[0], edges_MeV[-1]
    ])

    dt_fs = time_fs[-1]/steps[-1]

    def time2steps(x):
        return x / dt_fs

    def steps2time(x):
        return x * dt_fs

    EE,tt = np.meshgrid(edges_MeV,time_fs)

    fig,ax = plt.subplots(1, 1, figsize=(8,6), constrained_layout='True')

    im = ax.pcolormesh(
        tt,
        EE,
        data,
        norm=LogNorm(),
        rasterized=True
    )

    plt.colorbar(mappable=im,ax=ax,label=r"d$N$/d$\mathcal{E}$ (a.u.)")

    secax = ax.secondary_xaxis('top', functions=(time2steps, steps2time))
    secax.set_xlabel('steps')

    ax.set_ylim(ymin,ymax)

    ax.set_xlabel('time (fs)')
    ax.set_ylabel('kin. energy (MeV)')
    ax.set_title(f'kinetic energy of fw protons')

    # Construct the directory path
    out_dir = f'{sim_dir}/analysis/waterfall/{detector_str}/'
    # Name output file
    out_file = f'energy_waterfall_{sim_dir.split("/")[-1]}_{detector_str}.png'

    # Create the directory, including any necessary parent directories
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(f'{out_dir}/{out_file}',dpi=90)
    plt.close("all")

    return 0



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process a 1D proton energy histogram and give out a single number.')
    parser.add_argument("hist1D_file", help="The 1D histogram file to be analyzed. Bins are expected to be in normalized momentum units.")
    # Add optional arguments with default values
    parser.add_argument('--energy_MeV_lo', type=float, default=5.0, help='The lower bound of energy in MeV (default: 5.0)')
    parser.add_argument('--energy_MeV_hi', type=float, default=20.0, help='The upper bound of energy in MeV (default: 20.0)')

    args = parser.parse_args()

    path = './diags/reducedfiles/' + args.hist1D_file

    analyze_hist1D(filepath=path,
                   Ekin_MeV_lo=args.energy_MeV_lo,
                   Ekin_MeV_hi=args.energy_MeV_hi,
                   time_readout_fs=1100
                   )
