""" Laser pulse input creation script """

import lasy
from lasy.laser import Laser
from lasy.profiles import CombinedLongitudinalTransverseProfile, FromOpenPMDProfile
from lasy.profiles.longitudinal import LongitudinalProfileFromData, GaussianLongitudinalProfile
from lasy.profiles.transverse import GaussianTransverseProfile
import numpy as np
import scipy.constants as sc
from datetime import datetime

def create_laser_input(input_params):
    print("[PREPARATION] Current date and time:", datetime.now())
    wavelength = 815e-9  # Laser wavelength in meters
    pol = (1, 0)  # Linearly polarized in the x direction
    ratio_energy = 0.58  # Ratio of energy in focus
    laser_energy = ratio_energy * 20  # Energy of the laser pulse in joules
    waist = 2.12e-6  # Waist of the laser pulse in meters
    tau = 29.8e-15  # Pulse duration of the laser in seconds (i.e. 35 fs FWHM, tau=FWHM_I/1.1741)
    t_peak = 0.0  # Location of the peak of the laser pulse in time
    TOD = input_params["TOD_fs3"] * (1e-15) ** 3  # 80k fs^3
    alpha = 30  # angle (degrees) of laser incidence
    d_foc_z = 30.e-6  # distance from laser init plane to target surface in z (30 µm) which needs to be added to focal distance
    z_offset_foc_um = -input_params['z_pos_um']  # converts target position into laser focal offset in z
    d_foc = (d_foc_z + z_offset_foc_um * 1e-6) / np.cos(alpha * np.pi / 180)  # focal distance of the laser pulse
    time_window_fs = 1000

    dimensions = 'rt'  # Use cylindrical geometry
    lo = (0,-time_window_fs/2*1e-15)        # Lower bounds of the simulation box
    hi = (10*waist,time_window_fs/2*1e-15)  # Upper bounds of the simulation box
    # change this and the `lambda_bw_nm` below depending on the "noise level" needed
    # currently, the signal-to-noise ratio is 1e15
    num_points = (1000, 6000)  # Number of points in each dimension

    # Generate laser spectral intensity, including TOD, with numpy
    lambda_bw_nm = 200
    lambda_half_bw = lambda_bw_nm*1e-9/2
    lambda_range = np.linspace(wavelength-lambda_half_bw, wavelength+lambda_half_bw, num_points[-1])
    omega = 2 * np.pi * sc.c / lambda_range
    omega0 = 2 * np.pi * sc.c / wavelength
    intensity = np.exp(-(omega - omega0) ** 2 * tau ** 2 / 2)
    phase = TOD * (omega - omega0) ** 3 / 6  # From the definition of the TOD

    # Create corresponding laser profile, by combining
    # a longitudinal profile defined by the above spectral info
    # and a transverse Gaussian profile
    long_profile = LongitudinalProfileFromData(
        {'datatype': 'spectral',
         'axis': lambda_range,
         'intensity': intensity,
         'phase': phase,
         'dt': (hi[-1] - lo[-1]) / num_points[-1]},
        lo=lo[-1],
        hi=hi[-1])
    # focal distance from calculation above (expressed via target z-position and projection due to angle of incidence)
    trans_profile = GaussianTransverseProfile(waist, wavelength=wavelength, z_foc=d_foc)
    laser_profile = CombinedLongitudinalTransverseProfile(wavelength, pol, laser_energy, long_profile, trans_profile)

    # Define laser on a grid
    laser = Laser(dimensions, lo, hi, num_points, laser_profile)
    # Write laser profile out into
    laser.write_to_file('laser_with_tod', 'h5')

    return 0


def analyze_peak_from_file(work_dir='./', output_params = {}):
    """
    Analyze the peak intensity and its coordinates from a laser profile file.

    Parameters:
    - path_to_directory: str, optional
        The directory path where the laser profile file is located. Default is './'.
    - prefix: str, optional
        The prefix of the file name to identify the laser profile. Default is 'laser_with_tod'.

    Returns:
    - I_peak: float
        The peak intensity value from the laser profile.
    - peak_coords: numpy.ndarray
        The coordinates of the peak intensity in the laser profile.
    """

    # Create a laser profile object from an OpenPMD file using the given parameters
    laser_profile = FromOpenPMDProfile(
        path=work_dir,
        iteration=0,              # Specify the iteration number to read
        pol=(1,0),                # Polarization state of the laser
        theta=0.,                 # Angle of the plane of observation with respect to xz (xt)
        field='laserEnvelope',    # Specifies the field type to be read
        prefix='laser_with_tod',  # Prefix of the filename to read
        is_envelope=True,         # Indicates if the field is an envelope field
        verbose=True              # Enable verbose output for debugging
    )

    # Initialize lists to store lower and upper bounds for each axis
    los = []
    his = []

    # Iterate over the axes in the laser profile to get their bounds
    for key in laser_profile.axes:
        los.append(laser_profile.axes[key][0])   # Lower bound of the current axis
        his.append(laser_profile.axes[key][-1])  # Upper bound of the current axis

    # Create a Laser object using the profile and axis bounds
    laser = Laser(
        dim=laser_profile.dim,         # Dimensionality of the laser profile
        lo=los,                        # Lower bounds for each axis
        hi=his,                        # Upper bounds for each axis
        profile=laser_profile,         # The laser profile object
        npoints=laser_profile.array.shape # Number of points in the laser profile grid
    )

    # Calculate the intensity from the laser's field
    # np.squeeze is used to remove single-dimensional entries from the array
    intensity = np.squeeze(np.abs(laser.grid.field)**2 * sc.c * sc.epsilon_0 / 2)

    # Find the index of the maximum intensity value in the flattened array
    flat_idx = np.argmax(intensity)

    # Convert the flat index back to multi-dimensional indices
    idx = np.unravel_index(flat_idx, intensity.shape)

    # Get the peak intensity value at the calculated index
    I_peak = intensity[idx]

    # Calculate the physical coordinates of the peak intensity
    peak_coords = np.array(idx) * np.array(laser.grid.dx)

    output_params['intensity_peak'] = I_peak

    for key,max_in_key in zip(laser_profile.axes, peak_coords):
        output_params[f'peak_in_{key}'] = max_in_key

    return output_params  # Return the peak intensity and its coordinates


if __name__ == "__main__":

    # maybe superfluous but summarizes which parameters are replaced
    input_params = {
        "TOD_fs3" : {{TOD_fs3}},
        "z_pos_um" : {{z_pos_um}}
    }

    print("Creating laser pulse")
    create_laser_input(input_params=input_params)
    print("Laser pulse creation finished")
