Injector for staging experiment in BELLA PW
===========================================

This directory contains simulation data and analysis tools for injection experiments at the BELLA PW.

Each simulation starts by creating a LASY profile, then runs a WarpX simulation of the laser-plasma interaction and finally analyzes the results.

How to run
----------

Log on to Perlmutter (NERSC).

You need an environment that can run the WarpX code.
Refer to the `WarpX documentation <https://warpx.readthedocs.io/en/latest/install/hpc/perlmutter.html>`_ for setting this up.
Then install additional packages in the same environment:

.. code-block:: bash

   python -m pip install pymongo
   python -m pip install imageio[ffmpeg]
   python -m pip install git+https://github.com/RemiLehe/transparent_imshow
   python -m pip install git+https://github.com/openPMD/openPMD-viewer@get_laser_spectral_intensity

You will also need to compile WarpX with the following command:

.. code-block:: bash

   cd $HOME/src/warpx
   rm -rf build_pm_gpu
   cmake -S . -B build_pm_gpu -DWarpX_COMPUTE=CUDA -DWarpX_FFT=ON -DWarpX_DIMS="RZ"
   cmake --build build_pm_gpu -j 16

then copy the WarpX executable to `templates/warpx`.

You can then run a parameter scan with the following command:

.. code-block:: bash

   sbatch submission_script_multi
