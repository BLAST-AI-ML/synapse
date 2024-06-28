""" Study of TOD and target z-position in laser-ion acceleration for IFE Superfacilities LDRD
"""


from optimas.core import Parameter, VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator,FunctionEvaluator,ChainEvaluator
from optimas.explorations import Exploration

from analyze_histogram_1D import analyze_hist1D
from prepare_simulation import create_laser_input

# TODO remove at the end
# enable local testing environment
LOCAL_TESTING = True

# Create varying parameters and objectives.
var_1 = VaryingParameter("TOD_fs3", -80000, 80000)  # recommend doing 9 or 17 steps
var_2 = VaryingParameter("z_pos_um", -150, 150)  # script converts this to laser focal position
obj = Objective("f", minimize=False)  # the objective will be maximized (or would be, if an optimization were to be run)

n_TOD_vals = 9
n_zpos_vals = 9

# TODO remove at the end
if LOCAL_TESTING:
    n_TOD_vals = 3
    n_zpos_vals = 3

n_total = n_TOD_vals * n_zpos_vals

# Create generator.
gen = GridSamplingGenerator(
    varying_parameters=[var_1,var_2],
    objectives=[obj],
    n_steps=[n_TOD_vals,n_zpos_vals],
)

# Create evaluator.
# TODO need to make this a TemplateEvaluator because ChainEvaluators only support those
ev_pre = TemplateEvaluator(
    sim_template="prepare_simulation.py",  # this creates the lasy input files for the WarpX simulations
)

# TODO remove at the end
if not LOCAL_TESTING:
# Create evaluator.
    ev_main = TemplateEvaluator(
        sim_template="template_simulation_script",
        analysis_func=analyze_hist1D,
        executable="warpx.2d",
        n_gpus=32,
        env_script='/global/homes/m/mgarten/perlmutter_gpu_warpx.profile',  # point this to your WarpX profile,
        env_mpi='srun',  # dunno if that is really necessary ... potentially OPTIONAL,
    )

else:
    ev_main = TemplateEvaluator(
        sim_template="test_simulation_script_local",
        analysis_func=analyze_hist1D,
        executable="warpx.2d",
        n_gpus=1,
        env_script='/home/mgarten/warpx-gpu-mpich-dev.profile',  # for local testing purposes
        env_mpi='mpiexec',  # dunno if that is really necessary ... potentially OPTIONAL,
    )

# Create chain of evaluators
ev_chain = ChainEvaluator(
    evaluators=[ev_pre, ev_main]
)

# Create exploration.
exp = Exploration(
    generator=gen,
    evaluator=ev_chain,
    max_evals=n_total,
    sim_workers=4,  # 8 nodes per job currently, makes this a 32-node job (should be under 15min per run)
    run_async=True,  # with the GridSamplingGenerator it should not matter if we run in batches,
    libe_comms='mpi',
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()