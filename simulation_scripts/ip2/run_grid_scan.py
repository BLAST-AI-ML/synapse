""" Study of TOD and target z-position in laser-ion acceleration for IFE Superfacilities LDRD
"""


from optimas.core import Parameter, VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator,FunctionEvaluator,ChainEvaluator
from optimas.explorations import Exploration

from analyze_histogram_1D import analyze_hist1D



# Specify the analysis function.
def analysis_func_main(work_dir, output_params):
    num_particles = analyze_hist1D(
        filepath=f"{work_dir}/diags/reducedfiles/histuH_fw.txt",
        Ekin_MeV_lo=5,
        Ekin_MeV_hi=20,
        time_readout_fs=1100
    )
    output_params['f'] = num_particles

# Create varying parameters and objectives.
var_1 = VaryingParameter("TOD_fs3", -80000, 80000)  # recommend doing 9 or 17 steps
var_2 = VaryingParameter("z_pos_um", -150, 150)  # target z position - script converts this to laser focal position
obj = Objective("f", minimize=False)  # the objective will be maximized (or would be, if an optimization were to be run)

n_TOD_vals = 17
n_zpos_vals = 17
sim_workers = 16

n_total = n_TOD_vals * n_zpos_vals

# Create generator.
gen = GridSamplingGenerator(
    varying_parameters=[var_1,var_2],
    objectives=[obj],
    n_steps=[n_TOD_vals,n_zpos_vals],
    #analyzed_parameters=[par_1,par_2],
)

# Create evaluator.
ev_pre = TemplateEvaluator(
    sim_template="prepare_simulation.py",  # this creates the lasy input files for the WarpX simulations
    sim_files=[
        "warpx.2d",
        "analyze_histogram_1D.py",
        "template_inputs_2d",
        "input_spectral_intensity_scan005.csv",
        "input_spectral_intensity_scan006.csv"
    ],
)

# Create evaluator
ev_main = TemplateEvaluator(
    sim_template="template_inputs_2d",
    analysis_func=analysis_func_main,
    executable="warpx.2d",
    n_gpus=32,  # GPUs per individual evaluation
    env_script='/global/homes/m/mgarten/perlmutter_gpu_warpx.profile',  # point this to your WarpX profile,
    env_mpi='srun',  # dunno if that is really necessary ... potentially OPTIONAL,
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
    sim_workers=sim_workers,  # 8 nodes per job currently, makes this a 128-node job (should be under 15min per run)
    run_async=True,  # with the GridSamplingGenerator it should not matter if we run in batches,
)

# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
