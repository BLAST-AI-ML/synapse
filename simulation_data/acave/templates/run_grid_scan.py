""" Run parameter scan for A-cave experiment """
import os
from optimas.core import Parameter, VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator,FunctionEvaluator,ChainEvaluator
from optimas.explorations import Exploration

# Specify the analysis function.
def analysis_func_main(work_dir, output_params):
    output_params['f'] = 0

# Create varying parameters and objectives.
var_1 = VaryingParameter("target_to_focus_distance", -0.1, 0.1)
var_2 = VaryingParameter("fused_silica_thickness", -667, 667)
obj = Objective("f", minimize=False)

n_var_1 = 5
n_var_2 = 5
sim_workers = 5

n_total = n_var_1 * n_var_2

# Create generator.
gen = GridSamplingGenerator(
    varying_parameters=[var_1,var_2],
    objectives=[obj],
    n_steps=[n_var_1,n_var_2],
)

# Create evaluators
ev_pre = TemplateEvaluator(
    sim_template="templates/prepare_simulation.py",  # this creates the lasy input files for the WarpX simulations
    sim_files=[
        "templates/retrieval01_spectrum.csv"
    ],
    n_procs=1
)
ev_main = TemplateEvaluator(
    sim_template="templates/warpx_input_script",
    analysis_func=analysis_func_main,
    executable="templates/warpx.rz",
    n_gpus=12,  # GPUs per individual evaluation
    env_mpi='srun',  # dunno if that is really necessary ... potentially OPTIONAL,
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
save_dir = '/global/cfs/cdirs/m558/superfacility/simulation_data/acave/multi_' + os.environ['SLURM_JOB_ID']

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
