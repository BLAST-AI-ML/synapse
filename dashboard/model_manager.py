import asyncio
import numpy as np
from pathlib import Path
import re
from scipy.optimize import minimize
from sfapi_client import Client
from sfapi_client.compute import Machine
import sys
from lume_model.models.torch_model import TorchModel
from lume_model.models.gp_model import GPModel
from trame.widgets import vuetify2 as vuetify

from state_manager import state
from utils import load_model_file


class ModelManager:
    def __init__(self):
        print("Initializing model manager...")
        model_file = load_model_file()
        if model_file is None:
            self.__model = None
        else:
            # save model and model type
            self.__is_neural_network = False
            self.__is_gaussian_process = False
            try:
                if state.model_type == "Neural Network":
                    self.__is_neural_network = True
                    self.__model = TorchModel(model_file)
                elif state.model_type == "Gaussian Process":
                    self.__is_gaussian_process = True
                    self.__model = GPModel.from_yaml(model_file)
                else:
                    raise ValueError(f"Unsupported model type: {state.model_type}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                sys.exit(1)

    def avail(self):
        print("Checking model availability...")
        model_avail = True if self.__model is not None else False
        return model_avail

    @property
    def is_neural_network(self):
        return self.__is_neural_network

    @property
    def is_gaussian_process(self):
        return self.__is_gaussian_process

    def evaluate(self, parameters):
        print("Evaluating model...")
        if self.__model is not None:
            # evaluate model
            output_dict = self.__model.evaluate(parameters)
            if self.__is_neural_network:
                # expected only one value
                if len(output_dict.values()) != 1:
                    raise ValueError(
                        f"Expected 1 output value, but found {len(output_dict.values())}"
                    )
                # compute mean and mean error
                mean = list(output_dict.values())[0]
                mean_error = 0.0  # trick to collapse error range when lower/upper bounds are not predicted
            elif self.__is_gaussian_process:
                # TODO use "exp" only once experimental data is available for all experiments
                task_tag = "exp" if state.experiment == "ip2" else "sim"
                output_key = [key for key in output_dict.keys() if task_tag in key][0]
                # compute mean, standard deviation and mean error
                # (call detach method to detach gradients from tensors)
                mean = output_dict[output_key].mean.detach()
                std_dev = output_dict[output_key].variance.sqrt().detach()
                mean_error = 2.0 * std_dev
            else:
                raise ValueError(f"Unsupported model type: {state.model_type}")
            # compute lower/upper bounds for error range
            lower = mean - mean_error
            upper = mean + mean_error
            # convert to Python float if tensor has only one element
            # because Trame state variables must be serializable
            if mean.numel() == 1:
                mean = float(mean)
            return (mean, lower, upper)

    def model_wrapper(self, parameters_array):
        print("Wrapping model...")
        # convert array of parameters to dictionary
        parameters_dict = dict(zip(state.parameters.keys(), parameters_array))
        # change sign to the result in order to maximize when optimizing
        mean, lower, upper = self.evaluate(parameters_dict)
        res = -mean
        return res

    def optimize(self):
        # info print statement skipped to avoid redundancy
        if self.__model is not None:
            # get array of current parameters from state
            parameters_values = np.array(list(state.parameters.values()))
            # define parameters bounds for optimization
            parameters_bounds = []
            for key in state.parameters.keys():
                parameters_bounds.append(
                    (state.parameters_min[key], state.parameters_max[key])
                )
            # optimize model (maximize output value)
            res = minimize(
                fun=self.model_wrapper,
                x0=parameters_values,
                bounds=parameters_bounds,
            )
            # update parameters in state with optimal values
            state.parameters = dict(zip(state.parameters.keys(), res.x))
            # push again at flush time
            state.dirty("parameters")

    def get_output_transformers(self):
        print("Getting output transformers...")
        if self.__model is not None:
            return self.__model.output_transformers

    def training_kernel(self):
        try:
            # create an authenticated client
            with Client(
                client_id=state.sfapi_client_id, secret=state.sfapi_key
            ) as client:
                perlmutter = client.compute(Machine.perlmutter)
                # set the target path where auxiliary files will be copied
                target_path = "/global/cfs/cdirs/m558/superfacility/model_training/src/"
                [target_path] = perlmutter.ls(target_path, directory=True)
                # set the source path where auxiliary files are copied from
                source_path = Path.cwd().parent
                source_path_list = [
                    Path(source_path / "ml/NN_training/mongo_NN.py"),
                    Path(source_path / "ml/NN_training/Neural_Net_Classes.py"),
                    Path(source_path / "dashboard/config/variables.yml"),
                ]
                # copy auxiliary files to NERSC
                for path in source_path_list:
                    with open(path, "rb") as f:
                        f.filename = path.name
                        target_path.upload(f)
                # set the path of the script used to submit the training job on NERSC
                script_path = Path(
                    source_path / "automation/launch_model_training/training_pm.sbatch"
                )
                with open(script_path, "r") as file:
                    script_job = file.read()
                custom_arg = f"--experiment {state.experiment}"
                script_job = re.sub(
                    pattern=r"(srun python .*)",
                    repl=rf"\1 {custom_arg}",
                    string=script_job,
                )
                # submit the training job through the Superfacility API
                sfapi_job = perlmutter.submit_job(script_job)
                # print some logs
                print(f"Training job submitted (job ID: {sfapi_job.jobid})")
                # wait for the job to move into a terminal state
                sfapi_job.complete()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    async def training_async(self):
        try:
            print("Training model...")
            await asyncio.to_thread(self.training_kernel)
            print("Training job completed")
            # flush state and enable button
            state.model_training = False
            state.model_training_status = "Completed"
            state.flush()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def training_trigger(self):
        try:
            state.model_training = True
            state.model_training_status = "Submitted"
            state.flush()
            # schedule asynchronous job
            asyncio.create_task(self.training_async())
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def panel(self):
        print("Setting model card...")
        # list of available model types
        model_type_list = [
            "Gaussian Process",
            "Neural Network",
        ]
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_model", 0)):
            with vuetify.VExpansionPanel():
                vuetify.VExpansionPanelHeader(
                    "Control: Models", style="font-size: 20px; font-weight: 500;"
                )
                with vuetify.VExpansionPanelContent():
                    # create a row for the model selector
                    with vuetify.VRow():
                        vuetify.VSelect(
                            v_model=("model_type",),
                            items=("Models", model_type_list),
                            dense=True,
                            prepend_icon="mdi-brain",
                            style="margin-left: 16px; margin-top: 24px; max-width: 210px;",
                        )
                    # create a row for the switches and buttons
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VBtn(
                                "Train",
                                click=self.training_trigger,
                                disabled=("model_training",),
                                style="margin-left: 4px; margin-top: 12px; text-transform: none;",
                            )
                        with vuetify.VCol():
                            vuetify.VTextField(
                                v_model_number=("model_training_status",),
                                label="Training status",
                                readonly=True,
                                style="width: 100px;",
                            )
                        with vuetify.VCol():
                            vuetify.VSwitch(
                                v_model=("calibrate",),
                                label="Calibration",
                                inset=True,
                                style="margin-left: 16px;",
                            )
