# FY24/25 LDRD: IFE Superfacility

This is the project management for our FY24/25 LDRD on creating a blueprint for an IFE superfacility, that can combine experimental and simulation data through ML modeling with the goal to predict good experimental input parameters to achieve a certain goal (e.g., maximize number of particles produced in a certain energy range).

LDRD-funded partners: ATAP AMP (lead), ATAP BELLA (experiments), AMCRD (ML)

Self-funded partners: NERSC, SLAC


## Quick Links

- [Google Drive](https://drive.google.com/drive/u/0/folders/1bwManHU1j67kR008tj7KRuppZPCeaWuj)
- [Meeting Minutes](https://docs.google.com/document/d/1dcpWVORoMVZ1U-bFw1yFQzZOMu8ay4K9hkbavuiiwJI/edit)
- GChat: `Superfacility LDRD`
- Project Management:
  - [Milestones](https://github.com/ECP-WarpX/2024_IFE-superfacility/milestones)
  - [Tasks (Issues)](https://github.com/ECP-WarpX/2024_IFE-superfacility/issues)
  - [GitHub Team Access](https://github.com/orgs/ECP-WarpX/teams/ife-superfacility)
- Perlmutter (NERSC):
  - File directory: `/global/cfs/cdirs/m3239/ip2data`
  - Project: `m3239`
  - Unix Group: `ip2data`


## Organization of this repository

We will create separate folder to store the code associated with the different tasks:

- `simulation_data`: scripts that allow to produce simulations and extract a corresponding dataset, including WarpX and optimas script, submission scripts, etc. [(Tasks)](https://github.com/ECP-WarpX/2024_IFE-superfacility/milestone/1)
- `experimental_data`: scripts that extract datasets from the BELLA raw data. [(Tasks)](https://github.com/ECP-WarpX/2024_IFE-superfacility/milestone/2)
-  `ml`: scripts/notebook for experimentation with different ML models. [(Tasks)](https://github.com/ECP-WarpX/2024_IFE-superfacility/milestone/3)
-  `automation`: code that implements the orchestration workflow (automated copies, launching jobs at NERSC, etc.) [(Tasks)](https://github.com/ECP-WarpX/2024_IFE-superfacility/milestone/4)
