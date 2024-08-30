.. _examples-ip2-kapton:


Laser-Ion Acceleration at BELLA iP2 with Kapton Targets
=======================================================

A laser pulse with BELLA iP2 parameters impinges under 30 degrees onto a 13 micron Kapton foil to accelerate ions in the Target-Normal Sheath Acceleration scheme.
Simulations are executed in 2D via an `optimas <https://github.com/optimas-org/optimas>`_ grid scan.
Be aware that 2D simulations of laser-driven ion acceleration produce higher ion energies than in reality.
The drive laser pulse is created in a first step with the two input parameters for third order dispersion (TOD) and target z position (which gets translated to laser focal position).
WarpX simulations then read the laser pulse from an HDF5 file.
Analysis is directly performed and returns the number of protons between a lower and upper energy value as the objective function of the grid scan.

How to run
----------

Log on to Perlmutter (NERSC).
You require an environment that can run the WarpX code.
Refer to the `WarpX documentation <https://warpx.readthedocs.io/en/latest/install/hpc/perlmutter.html>`_ for setting this up.
Then add optimas from `its documentation <https://optimas.readthedocs.io/en/latest/user_guide/installation_perlmutter.html>`_ to that environment.
It is useful to have a helper file for loading the environment, e.g. a version of ``perlmutter_gpu_warpx.profile``, that sets the project to ``m3239_g``.
With the environment loaded run

.. code-block:: bash

    sbatch perlmutter_gpu_run_grid_scan.sbatch

either in a directory in ``$PSCRATCH`` or ``$CFS/m3239/ip2data/simulations/``.
``$PSCRATCH`` is usually preferred for production runs but it does not offer to share data in place.
Running in ``$CFS`` is possible but occasional write errors have been observed.
We currently choose to run in ``$PSCRATCH`` and transfer the simulations post-run to ``CFS`` via Globus.
To extract results from the history file into its own `simulation_results.csv` file, run:

.. code-block:: bash

    python write_results_to_csv.py


Outlook
-------

Possible updates in ``template_inputs_2d`` for future studies:

- CH surface layer (currently, protons come directly from the target)
- a realistic laser spectral phase

