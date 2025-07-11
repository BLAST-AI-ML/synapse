#!/usr/bin/env python3

import matplotlib.pyplot as plt
from lasy.laser import Laser
from lasy.profiles.transverse import JincTransverseProfile
from lasy.profiles.longitudinal import GaussianLongitudinalProfile
from lasy.profiles import CombinedLongitudinalTransverseProfile
import numpy as np

try:
    from mip4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0

def create_laser_pulse():

    # Laser parameters
    laser_energy = 12 # J
    w0 = 52.0e-6/2.66 # m
    tau = 35e-15/np.sqrt(2*np.log(2)) # s
    wavelength = 0.8e-6 # m
    target_to_focus_distance = 1e-2 # m

    tranverseProfile = JincTransverseProfile(w0)
    longitudinalProfile = GaussianLongitudinalProfile(
        wavelength=0.8e-6,  # m
        tau=tau,  # s
        t_peak=0.,  # s,
    )
    profile = CombinedLongitudinalTransverseProfile(
        wavelength=0.8e-6,  # m
        pol=(1, 0),
        laser_energy=1.,  # J
        long_profile = longitudinalProfile,
        trans_profile = tranverseProfile,
    )

    # Create laser with given profile in `rt` geometry.
    laser = Laser(
        dim="rt",
        lo=(0e-6, -2.5*tau),
        hi=(500e-6, +2.5*tau),
        npoints=(1000, 750),
        profile=profile,
        n_azimuthal_modes = 1,
    )

    laser.propagate(-target_to_focus_distance)

    laser.write_to_file(file_prefix="lasy_profile", file_format="h5")

if __name__ == '__main__':

    # When this script is called in optimas
    # it might be run by multiple processes
    # but we only need to run it once
    if rank == 0:
        create_laser_pulse()
