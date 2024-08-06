.. _examples-ip2-kapton:

Laser-Ion Acceleration at BELLA iP2 with Kapton Targets
=======================================================

A laser pulse with BELLA iP2 parameters impinges under 30 degrees onto a 13 micron Kapton foil to accelerate ions in the Target-Normal Sheath Acceleration scheme.
Simulations are executed in 2D via an `optimas <https://github.com/optimas-org/optimas>`_ grid scan.
Be aware that 2D simulations of laser-driven ion acceleration produce higher ion energies than in reality.
The drive laser pulse is created in a first step with the two input parameters for third order dispersion (TOD) and target z position (which gets translated to laser focal position).
WarpX simulations then read the laser pulse from an HDF5 file.
Analysis is directly performed and returns the number of protons between a lower and upper energy value as the objective function of the grid scan.

Possible updates in ``template_inputs_2d`` for future studies:

- CH surface layer (currently, protons come directly from the target)
- a realistic laser spectral phase

