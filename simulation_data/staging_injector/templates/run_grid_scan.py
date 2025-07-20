""" Run parameter scan for A-cave experiment """

from optimas.core import VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator, ChainEvaluator
from optimas.explorations import Exploration

# Specify the analysis function.
def analysis_func_main(work_dir, output_params):
    output_params['f'] = 0

# Create varying parameters and objectives.
# TODO: automatically read config file
var_1 = VaryingParameter("Laser energy [J]", 12, 17)
var_2 = VaryingParameter("Target-to-focus distance [cm]", 0, 2)
var_3 = VaryingParameter("Dopant concentration [%]", 0, 10)
var_4 = VaryingParameter("Background density [1e18/cm^3]", 3, 5)
obj = Objective("f", minimize=False)

n_steps = [2, 2, 2, 2]
sim_workers = 8

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
    sim_template="../templates/prepare_simulation.py",  # this creates the lasy input files for the WarpX simulations
    n_procs=1
)
ev_main = TemplateEvaluator(
    sim_template="../templates/inputs",
    analysis_func=analysis_func_main,
    executable="../templates/warpx",
    n_gpus=1,  # GPUs per individual evaluation
    env_mpi='srun',
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
    max_evals=n_total,
    sim_workers=sim_workers,
    run_async=True,  # with the GridSamplingGenerator it should not matter if we run in batches,
)

# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
