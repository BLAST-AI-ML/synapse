""" Laser pulse input creation script """

import lasy
from lasy.laser import Laser
from lasy.profiles import CombinedLongitudinalTransverseProfile
from lasy.profiles.longitudinal import LongitudinalProfileFromData, GaussianLongitudinalProfile
from lasy.profiles.transverse import GaussianTransverseProfile
import numpy as np
from scipy.constants import c

def create_laser_input(input_params, output_params):
    wavelength = 815e-9  # Laser wavelength in meters
    pol = (1, 0)  # Linearly polarized in the x direction
    laser_energy = 20  # Energy of the laser pulse in joules
    waist = 2.12e-6  # Waist of the laser pulse in meters
    tau = 29.8e-15  # Pulse duration of the laser in seconds (i.e. 35 fs FWHM, tau=FWHM_I/1.1741)
    t_peak = 0.0  # Location of the peak of the laser pulse in time
    TOD = input_params["TOD_fs3"] * (1e-15) ** 3  # 80k fs^3
    alpha = 30  # angle (degrees) of laser incidence
    d_foc_0 = 22.205039999826518e-6  # focal distance from laser init plane to target surface for 20 µm distance in z
    d_foc = d_foc_0 + input_params['z_pos_um'] * 1e-6 / np.cos(alpha * np.pi / 180)  # focal distance of the laser pulse

    dimensions = 'rt'  # Use cylindrical geometry
    # TODO we might need to start earlier than -3 * tau and this should probably depend on the TOD we are choosing
    # TODO we might want to check that the energy content in the created laser pulse is indeed what we expect
    lo = (0, -3 * tau)  # Lower bounds of the simulation box
    hi = (5 * waist, 12 * tau)  # Upper bounds of the simulation box
    # TODO figure out if I need to change this
    num_points = (300, 1500)  # Number of points in each dimension

    # Generate laser spectral intensity, including TOD, with numpy
    lambda_range = np.linspace(765e-9, 865e-9, num_points[-1])
    omega = 2 * np.pi * c / lambda_range
    omega0 = 2 * np.pi * c / wavelength
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

    t_peak = analyze_t_peak_from_file()
    # temporal offset of the laser pulse peak intensity due to TOD
    output_params['t_peak'] = t_peak

    return output_params


def analyze_t_peak_from_file():
    # TODO fill with analysis
    return np.float32(0.)


if __name__ == "__main__":

# maybe superfluous but summarizes which parameters are replaced
    input_params = {
        "TOD_fs3" : {{TOD_fs3}},
        "z_pos_um" : {{z_pos_um}}
    }

    output_params = {
        "t_peak" : 0
    }

    create_laser_input(input_params=input_params, output_params=output_params)
