"""Run parameter scan for A-cave experiment"""

import numpy as np
import os
from optimas.core import VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator, ChainEvaluator
from optimas.explorations import Exploration

import yaml
import argparse

# Parse arguments to determine whether to run a single simulation or a grid scan
parser = argparse.ArgumentParser()
parser.add_argument(
    "--single-simulation-parameters",
    default=None,
    help="Path to the YAML file containing simulation parameters ; if not provided, a grid scan will be run",
)
args = parser.parse_args()


# Specify the analysis function.
def analysis_func_main(work_dir, output_params):
    output_params["f"] = 0


# Create varying parameters and objectives.
parameters_list = [
    {"name": "laser_energy", "lower_bound": 12.5, "upper_bound": 13.5},
    {"name": "target_to_focus_distance", "lower_bound": 0, "upper_bound": 1},
    {"name": "dopant_concentration", "lower_bound": 0, "upper_bound": 3},
    {"name": "upstream_density", "lower_bound": 0.9, "upper_bound": 1.54},
    {"name": "downstream_density", "lower_bound": 0.33, "upper_bound": 1.1},
]
if args.single_simulation_parameters is None:
    varying_parameters = [
        VaryingParameter(
            name=param["name"],
            lower_bound=param["lower_bound"],
            upper_bound=param["upper_bound"],
        )
        for param in parameters_list
    ]
    # Number of steps for each varying parameter
    n_steps = [3, 5, 4, 6, 6]
    sim_workers = 240
else:
    # Read the simulation parameters from the input YAML file
    with open(args.single_simulation_parameters, "r") as f:
        single_simulation_parameters_dict = yaml.safe_load(f)
    # Define varying parameters with identical lower and upper bounds
    varying_parameters = [
        VaryingParameter(
            name=param["name"],
            lower_bound=single_simulation_parameters_dict[param["name"]],
            upper_bound=single_simulation_parameters_dict[param["name"]],
        )
        for param in parameters_list
    ]
    # Only one step for each varying parameter
    n_steps = [1, 1, 1, 1, 1]
    sim_workers = 1

obj = Objective("f", minimize=False)

# Compute total number of steps
n_total = np.prod(n_steps)

# Create generator
gen = GridSamplingGenerator(
    varying_parameters,
    objectives=[obj],
    n_steps=n_steps,
)

# Create evaluators
ev_pre = TemplateEvaluator(
    sim_template="templates/prepare_simulation.py",  # this creates the lasy input files for the WarpX simulations
    n_procs=1,
)
ev_main = TemplateEvaluator(
    sim_template="templates/warpx_input_script",
    analysis_func=analysis_func_main,
    executable="templates/warpx",
    n_gpus=1,  # GPUs per individual evaluation
    env_mpi="srun",
)
ev_post = TemplateEvaluator(sim_template="templates/analyze_simulation.py", n_procs=1)

# Create chain of evaluators
ev_chain = ChainEvaluator(evaluators=[ev_pre, ev_main, ev_post])

# Save simulation results in the shared folder, in a subfolder with the job id
save_dir_prefix = "single" if args.single_simulation_parameters else "multi"
slurm_job_id = os.environ["SLURM_JOB_ID"]
save_dir = f"/global/cfs/cdirs/m558/superfacility/simulation_data/staging_injector/{save_dir_prefix}_{slurm_job_id}"

# Create exploration
exp = Exploration(
    generator=gen,
    evaluator=ev_chain,
    max_evals=n_total,
    sim_workers=sim_workers,
    run_async=True,  # with the GridSamplingGenerator it should not matter if we run in batches,
    exploration_dir_path=save_dir,
)

# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
