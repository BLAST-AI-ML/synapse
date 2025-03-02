""" Run parameter scan for A-cave experiment """

from optimas.core import Parameter, VaryingParameter, Objective
from optimas.generators import GridSamplingGenerator
from optimas.evaluators import TemplateEvaluator,FunctionEvaluator,ChainEvaluator
from optimas.explorations import Exploration

# Specify the analysis function.
def analysis_func_main(work_dir, output_params):
    output_params['kHz_thorlab_spectrometer mean_wavelength'] = 0

# Create varying parameters and objectives.
var_1 = VaryingParameter("kHz_Hexapod_Target ypos", -0.1, 0.1)
var_2 = VaryingParameter("kHz_Zaber_Compressor Position.Ch1", 50e3, 100e3)
obj = Objective("kHz_thorlab_spectrometer mean_wavelength", minimize=False)

n_var_1 = 2
n_var_2 = 2
sim_workers = 2

n_total = n_var_1 * n_var_2

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
)
ev_main = TemplateEvaluator(
    sim_template="../templates/warpx_input_script",
    analysis_func=analysis_func_main,
    executable="../templates/warpx.rz",
    n_gpus=12,  # GPUs per individual evaluation
)
ev_post = TemplateEvaluator(
    sim_template="../templates/analyze_simulation.py",
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
