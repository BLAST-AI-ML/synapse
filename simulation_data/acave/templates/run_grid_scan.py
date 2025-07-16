""" Run parameter scan for A-cave experiment """

from optimas.core import VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator,ChainEvaluator
from optimas.explorations import Exploration

import pandas as pd
import argparse

# Parse arguments to determine whether to run a single simulation or a grid scan
parser = argparse.ArgumentParser()
parser.add_argument('--single-simulation-parameters', default=None,
    help='Path to the CSV file containing simulation parameters ; if not provided, a grid scan will be run')
args = parser.parse_args()

if args.single_simulation_parameters is None:
    # Grid scan with parallel simulations
    n_var_1 = 5
    n_var_2 = 5
    sim_workers = 5
    var_1 = VaryingParameter("target_to_focus_distance", -0.1, 0.1)
    var_2 = VaryingParameter("fused_silica_thickness", -667, 667)
else:
    # Single simulation, with parameters provided in the CSV file
    n_var_1 = 1
    n_var_2 = 1
    sim_workers = 1
    # Extract parameters from the CSV file
    df = pd.read_csv(args.single_simulation_parameters)
    # Set the varying parameters with the same min and max value,
    # so that the simulation is run exactly with the provided parameters
    var_1 = VaryingParameter("target_to_focus_distance", df["sim_val"].iloc[0], df["sim_val"].iloc[1])
    var_2 = VaryingParameter("fused_silica_thickness", df["fused_silica_thickness"].iloc[0], df["fused_silica_thickness"].iloc[0])
    
# Specify the analysis function.
def analysis_func_main(work_dir, output_params):
    output_params['f'] = 0

# Create varying parameters and objectives.
obj = Objective("f", minimize=False)


# Create generator.
gen = GridSamplingGenerator(
    varying_parameters=[var_1,var_2],
    objectives=[obj],
    n_steps=[n_var_1,n_var_2],
)

# Create evaluators
ev_pre = TemplateEvaluator(
    sim_template="../templates/prepare_simulation.py",  # this creates the lasy input files for the WarpX simulations
    sim_files=[
        "../templates/retrieval01_spectrum.csv"
    ],
    n_procs=1
)
ev_main = TemplateEvaluator(
    sim_template="../templates/warpx_input_script",
    analysis_func=analysis_func_main,
    executable="../templates/warpx.rz",
    n_gpus=12,  # GPUs per individual evaluation
    env_mpi='srun',  # dunno if that is really necessary ... potentially OPTIONAL,
)
ev_post = TemplateEvaluator(
    sim_template="../templates/analyze_simulation.py",
    n_procs=1
)

# Create chain of evaluators
ev_chain = ChainEvaluator(
    evaluators=[ev_pre, ev_main, ev_post]
)

# Create exploration.
exp = Exploration(
    generator=gen,
    evaluator=ev_chain,
    max_evals=n_var_1 * n_var_2,
    sim_workers=sim_workers,
    run_async=True,  # with the GridSamplingGenerator it should not matter if we run in batches,
)

# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
