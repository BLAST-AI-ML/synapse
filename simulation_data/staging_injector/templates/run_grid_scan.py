"""Run parameter scan for A-cave experiment"""

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
# TODO Use a parameter list to make this more compact
if args.single_simulation_parameters is None:
    var_1 = VaryingParameter("laser_energy", 12.5, 13.5)
    var_2 = VaryingParameter("target_to_focus_distance", 0, 1)
    var_3 = VaryingParameter("dopant_concentration", 0, 3)
    var_4 = VaryingParameter("upstream_density", 0.9, 1.54)
    var_5 = VaryingParameter("downstream_density", 0.33, 1.1)
    # Number of steps for each varying parameter
    n_steps = [3, 5, 4, 6, 6]
    sim_workers = 240
else:
    # Read the simulation parameters from the input YAML file
    with open(args.single_simulation_parameters, "r") as f:
        single_simulation_parameters_dict = yaml.safe_load(f)
    # Define varying parameters with identical lower and upper bounds
    var_name = "laser_energy"
    var_1 = VaryingParameter(
        name=var_name,
        lower_bound=single_simulation_parameters_dict[var_name],
        upper_bound=single_simulation_parameters_dict[var_name],
    )
    var_name = "target_to_focus_distance"
    var_2 = VaryingParameter(
        name=var_name,
        lower_bound=single_simulation_parameters_dict[var_name],
        upper_bound=single_simulation_parameters_dict[var_name],
    )
    var_name = "dopant_concentration"
    var_3 = VaryingParameter(
        name=var_name,
        lower_bound=single_simulation_parameters_dict[var_name],
        upper_bound=single_simulation_parameters_dict[var_name],
    )
    var_name = "upstream_density"
    var_4 = VaryingParameter(
        name=var_name,
        lower_bound=single_simulation_parameters_dict[var_name],
        upper_bound=single_simulation_parameters_dict[var_name],
    )
    var_name = "downstream_density"
    var_5 = VaryingParameter(
        name=var_name,
        lower_bound=single_simulation_parameters_dict[var_name],
        upper_bound=single_simulation_parameters_dict[var_name],
    )
    # Only one step for each varying parameter
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
save_dir = (
    f"/global/cfs/cdirs/m558/superfacility/simulation_data/staging_injector/{save_dir_prefix}_"
    + os.environ["SLURM_JOB_ID"]
)

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
