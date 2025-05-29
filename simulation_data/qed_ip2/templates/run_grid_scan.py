""" Run parameter scan for A-cave experiment """

from optimas.core import Parameter, VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator,FunctionEvaluator,ChainEvaluator
from optimas.explorations import Exploration

# Specify the analysis function.
def analysis_func_main(work_dir, output_params):
    output_params['f'] = 0

# Create varying parameters and objectives.
var_1 = VaryingParameter("prepulse_delay", 67.41, 71.41)
var_2 = VaryingParameter("target_focus", -56.81, -56.87)
obj = Objective("f", minimize=False)

n_var_1 = 8
n_var_2 = 5
sim_workers = 5

n_total = n_var_1 * n_var_2

# Create generator.
gen = GridSamplingGenerator(
    varying_parameters=[var_1,var_2],
    objectives=[obj],
    n_steps=[n_var_1,n_var_2],
)

ev_main = TemplateEvaluator(
    sim_template="../templates/warpx_input_script",
    analysis_func=analysis_func_main,
    executable="../templates/warpx",
    n_gpus=16,  # GPUs per individual evaluation
    env_mpi='srun',  # dunno if that is really necessary ... potentially OPTIONAL,
)
ev_post = TemplateEvaluator(
    sim_template="../templates/analyze_simulation.py",
    n_procs=1
)

# Create chain of evaluators
ev_chain = ChainEvaluator(
    evaluators=[ev_main, ev_post]
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
