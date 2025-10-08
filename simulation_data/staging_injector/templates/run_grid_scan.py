"""Run parameter scan for A-cave experiment"""

import os
from optimas.core import VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator, ChainEvaluator
from optimas.explorations import Exploration

import yaml
import pandas as pd
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
if args.single_simulation_parameters is None:
    var_1 = VaryingParameter("laser_energy", 12.5, 13.5)
    var_2 = VaryingParameter("target_to_focus_distance", 0, 1)
    var_3 = VaryingParameter("dopant_concentration", 0, 3)
    var_4 = VaryingParameter("upstream_density", 0.9, 1.54)
    var_5 = VaryingParameter("downstream_density", 0.33, 1.1)

    n_steps = [3, 5, 4, 6, 6]
    sim_workers = 240
else:
    # Read the simulation parameters from the input YAML file
    with open(args.single_simulation_parameters, "r") as f:
        yaml_data = yaml.safe_load(f)
    df = pd.DataFrame(yaml_data.items(), columns=["exp_name", "sim_val"])

    # FIXME Avoid hardcoding the order of parameters
    var_1 = VaryingParameter(
        "laser_energy", df["sim_val"].iloc[0], df["sim_val"].iloc[0]
    )
    var_2 = VaryingParameter(
        "target_to_focus_distance", df["sim_val"].iloc[1], df["sim_val"].iloc[1]
    )
    var_3 = VaryingParameter(
        "dopant_concentration", df["sim_val"].iloc[2], df["sim_val"].iloc[2]
    )
    var_4 = VaryingParameter(
        "upstream_density", df["sim_val"].iloc[3], df["sim_val"].iloc[3]
    )
    var_5 = VaryingParameter(
        "downstream_density", df["sim_val"].iloc[4], df["sim_val"].iloc[4]
    )

    n_steps = [1, 1, 1, 1, 1]
    sim_workers = 1

obj = Objective("f", minimize=False)

# Compute total number of steps
n_total = 1
for n_step in n_steps:
    n_total *= n_step

# Create generator
gen = GridSamplingGenerator(
    varying_parameters=[var_1, var_2, var_3, var_4, var_5],
    objectives=[obj],
    n_steps=n_steps,
)

# Create evaluators
ev_pre = TemplateEvaluator(
    sim_template="../templates/prepare_simulation.py",  # this creates the lasy input files for the WarpX simulations
    n_procs=1,
)
ev_main = TemplateEvaluator(
    sim_template="templates/inputs",
    analysis_func=analysis_func_main,
    executable="../templates/warpx",
    n_gpus=1,  # GPUs per individual evaluation
    env_mpi="srun",
)
ev_post = TemplateEvaluator(sim_template="templates/analyze_simulation.py", n_procs=1)

# Create chain of evaluators
ev_chain = ChainEvaluator(evaluators=[ev_pre, ev_main, ev_post])

# Create exploration.

# Save simulation results in the shared folder, in a subfolder with the job id
save_dir = (
    "/global/cfs/cdirs/m558/superfacility/simulation_data/staging_injector/multi_"
    + os.environ["SLURM_JOB_ID"]
)

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
