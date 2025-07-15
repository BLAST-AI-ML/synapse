Nonlinear laser propagation at BELLA A-Cave
=========================================

This directory contains simulation data and analysis tools for laser-plasma interaction in the A-Cave BELLA setup.
The simulations are designed to study the nonlinear laser propagation, including ionization defocusing and laser red-shifting from the laser-plasma interaction.

Each simulation starts by creating a LASY profile (which includes propagation of the laser through fused silica),
then runs a WarpX simulation of the laser-plasma interaction and finally analyzes the results
(esp. extracts the mean wavelength of the laser after the interaction).

How to run
----------

Log on to Perlmutter (NERSC).
You need an environment that can run the WarpX code.
Refer to the `WarpX documentation <https://warpx.readthedocs.io/en/latest/install/hpc/perlmutter.html>`_ for setting this up.
Then install `openPMD-viewer` and `pymongo` in the same environment:

.. code-block:: bash

   python -m pip install pymongo
   python -m pip install git+https://github.com/openPMD/openPMD-viewer@get_laser_spectral_intensity

WarpX should be compiled with this command:

.. code-block::bash

   cd $HOME/src/warpx
   rm -rf build_pm_gpu

   cmake -S . -B build_pm_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_DIMS="RZ"
   cmake --build build_pm_gpu -j 16

To run simulations, copy (or clone) the whole folder `simulation_data/acave` into your `$SCRATCH` folder.
Then copy the compiled WarpX executable to `simulation_data/acave/templates/warpx.rz`.

Single simulations can be run with

.. code-block:: bash

   sbatch submission_script_oneoff

Scans of parameters can be run with

.. code-block:: bash

   sbatch submission_script_multi
