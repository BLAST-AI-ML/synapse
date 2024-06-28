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
import numpy as np
import pandas as pd
import scipy.constants as sc

def analyze_hist1D(
        filepath=f'./diags/reducedfiles/histuH.txt',
        Ekin_MeV_lo=5, Ekin_MeV_hi=20,
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
    df['time_difference'] = (df['1'] - time_readout).abs()

    # Find the row with the minimum difference
    closest_entry = df.loc[df['time_difference'].idxmin()]

    bins_data = closest_entry.iloc[2:1002].values

    num_in_interval = np.sum(bins_data[my_filter])
    print(num_in_interval)

    return num_in_interval

def plot_histogram_waterfall():
    """
    TODO
    :return: None
    """

    return 0



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process a 1D proton energy histogram and give out a single number.')
    parser.add_argument("hist1D_file", help="The 1D histogram file to be analyzed. Bins are expected to be in normalized momentum units.")
    # Add optional arguments with default values
    parser.add_argument('--energy_MeV_lo', type=float, default=0.0, help='The lower bound of energy in MeV (default: 0.0)')
    parser.add_argument('--energy_MeV_hi', type=float, default=100.0, help='The upper bound of energy in MeV (default: 100.0)')

    args = parser.parse_args()

    path = './diags/reducedfiles/' + args.hist1D_file

    analyze_hist1D(filepath=path,
                   Ekin_MeV_lo=args.energy_MeV_lo,
                   Ekin_MeV_hi=args.energy_MeV_hi,
                   time_readout_fs=600
                   )