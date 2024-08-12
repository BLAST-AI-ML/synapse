This folder contains files that illustrate how to automate workflows.

# Setting up your environment

## Install dependencies

```
pip install -r requirements.txt
```

## Get credentials for NERSC Superfacility API

As documented [here](https://docs.nersc.gov/services/sfapi/#getting-started), connect to
[iris.nersc.gov/profile](https://iris.nersc.gov/profile) and:
    - click the upper right icon with your username
    - scroll down to the section `Superfacility API Client`, at the bottom of the page
    - click `New Client`
    - enter a client name (e.g. `Test`), move the security level slider to Red, and select `Your IP` in the `IP Presets` menu
    - click the button `Copy` next to **New Client Id**
    - click the button `Download` next to **Your Private Key (PEM format)**
    - open the downloaded file (`priv_key.pem`) and copy the new client Id on the first line.

Then move the file `priv_key.pem` to the folder containing this README file and run `chmod 600 priv_key.pem` on the Terminal.
