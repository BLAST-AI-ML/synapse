This folder contains files that illustrate how to launch WarpX simulations at NERSC, from a local computer.

In order for this to run properly, you need to setup your environment at NERSC and on your local computer.

# Setting up your environment at NERSC

- Install WarpX and its dependencies as described [here](https://warpx.readthedocs.io/en/latest/install/hpc/perlmutter.html).
  In the [Compilation](https://warpx.readthedocs.io/en/latest/install/hpc/perlmutter.html#compilation) section, use the first set of instructions
  (i.e. the ones for the executable, as opposed to the Python instructions):

```
cd $HOME/src/warpx
rm -rf build_pm_gpu

cmake -S . -B build_pm_gpu -DWarpX_COMPUTE=CUDA -DWarpX_DIMS="RZ"
cmake --build build_pm_gpu -j 16
```

- Copy the folder `templates` from this Github repo to `$SCRATCH/warpx_ip2/templates`.
  (Create the intermediate folder `warpx_ip2` with `mkdir` if needed)

- Copy the warpx executable to that folder:
```
cp $HOME/src/warpx/build_pm_gpu/bin/warpx.2d $SCRATCH/warpx_ip2/templates
```

# Setting up your environment on your local computer

## Install dependencies

```
conda env create -f environment.yml
```

```
conda activate sfapi
```

## Get credentials for NERSC Superfacility API

As documented [here](https://docs.nersc.gov/services/sfapi/#getting-started), connect to
[iris.nersc.gov/profile](https://iris.nersc.gov/profile) and:

- click the upper right icon with your username
- scroll down to the section `Superfacility API Client`, at the bottom of the page
- click `New Client`
- enter a client name (e.g. `Test`), move the security level slider to **Red**, and select `Your IP` in the `IP Presets` menu
- click the button `Copy` next to **New Client Id**
- click the button `Download` next to **Your Private Key (PEM format)**
- open the downloaded file (`priv_key.pem`) and copy the new client Id on the first line.

Then move the file `priv_key.pem` to the folder containing this README file and run `chmod 600 priv_key.pem` on the Terminal.

## Login to prefect

- Create a new account or sign in at [app.prefect.cloud/](Create a new account or sign in at https://app.prefect.cloud/.)
- From the Terminal, run
```
prefect cloud login
```
Choose Log in with a web browser and click the Authorize button in the browser window that opens.
