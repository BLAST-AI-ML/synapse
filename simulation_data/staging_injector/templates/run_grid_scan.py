""" Run parameter scan for A-cave experiment """
import os
from optimas.core import VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator, ChainEvaluator
from optimas.explorations import Exploration

# Specify the analysis function.
def analysis_func_main(work_dir, output_params):
    output_params['f'] = 0

# Create varying parameters and objectives.
var_1 = VaryingParameter("laser_energy", 12, 17)
var_2 = VaryingParameter("target_to_focus_distance", 0, 2)
var_3 = VaryingParameter("dopant_concentration", 0, 10)
var_4 = VaryingParameter("background_density", 3, 5)
obj = Objective("f", minimize=False)

n_steps = [5, 7, 5, 7]
sim_workers = 240

# Compute total number of steps
n_total = 1
for n_step in n_steps:
    n_total *= n_step

# Create generator
gen = GridSamplingGenerator(
    varying_parameters=[var_1,var_2,var_3,var_4],
    objectives=[obj],
    n_steps=n_steps,
)

# Create evaluators
ev_pre = TemplateEvaluator(
    sim_template="templates/prepare_simulation.py",  # this creates the lasy input files for the WarpX simulations
    n_procs=1
)
ev_main = TemplateEvaluator(
    sim_template="templates/inputs",
    analysis_func=analysis_func_main,
    executable="templates/warpx",
    n_gpus=1,  # GPUs per individual evaluation
    env_mpi='srun',
)
ev_post = TemplateEvaluator(
    sim_template="templates/analyze_simulation.py",
    n_procs=1
)

# Create chain of evaluators
ev_chain = ChainEvaluator(
    evaluators=[ev_pre, ev_main, ev_post]
)

# Create exploration.

# Save simulation results in the shared folder, in a subfolder with the job id
save_dir = '/global/cfs/cdirs/m558/superfacility/simulation_data/staging_injector/multi_' + os.environ['SLURM_JOB_ID']

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
