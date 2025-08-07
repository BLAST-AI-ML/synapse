import asyncio
import numpy as np
from pathlib import Path
import tempfile
import os
import yaml
import re
from scipy.optimize import minimize
from sfapi_client import AsyncClient
from sfapi_client.compute import Machine
from lume_model.models.torch_model import TorchModel
from lume_model.models.gp_model import GPModel
from trame.widgets import vuetify2 as vuetify
from utils import load_config_file, metadata_match
from error_manager import add_error
from datetime import datetime
from sfapi_manager import monitor_sfapi_job

from state_manager import state

model_type_tag_dict = {
    "Gaussian Process": "GP",
    "Neural Network": "NN",
}


class ModelManager:
    def __init__(self, db):
        print("Initializing model manager...")
        # Set initial default values
        self.__model = None
        self.__is_neural_network = False
        self.__is_gaussian_process = False

        # Download model information from the database
        collection = db["models"]
        model_type_tag = model_type_tag_dict[state.model_type]
        query = {"experiment": state.experiment, "model_type": model_type_tag}
        count = collection.count_documents(query)
        if count == 0:
            print(
                f"No model found for experiment: {state.experiment} and model type: {model_type_tag}"
            )
            return
        elif count > 1:
            print(
                f"Multiple models found ({count}) for experiment: {state.experiment} and model type: {model_type_tag}!"
            )
            return

        # Load model information from the database
        document = collection.find_one(query)
        # Save model files in a temporary directory,
        # so that it can then be loaded with lume_model
        with tempfile.TemporaryDirectory() as temp_dir:
            # - Save the model yaml file
            yaml_file_content = document["yaml_file_content"]
            with open(os.path.join(temp_dir, state.experiment + ".yml"), "w") as f:
                f.write(yaml_file_content)
            # - Save the corresponding binary files
            model_info = yaml.safe_load(yaml_file_content)
            filenames = (
                [model_info["model"]]
                + model_info["input_transformers"]
                + model_info["output_transformers"]
            )
            for filename in filenames:
                with open(os.path.join(temp_dir, filename), "wb") as f:
                    f.write(document[filename])

            # Check consistency of the model file
            print("Reading model file...")
            config_file = load_config_file()
            model_file = os.path.join(temp_dir, f"{state.experiment}.yml")
            if not os.path.isfile(model_file):
                print(f"Model file {model_file} not found")
                return
            elif not metadata_match(config_file, model_file):
                print(
                    f"Model file {model_file} does not match configuration file {config_file}"
                )
                return

            # Load model with lume_model
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
                title = f"Unable to load model {state.model_type}"
                msg = f"Error occurred when loading model: {e}"
                add_error(title, msg)
                print(msg)

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
                # compute mean and mean error
                mean = output_dict[state.displayed_output]
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

    async def training_kernel(self):
        try:
            # create an authenticated client
            async with AsyncClient(
                client_id=state.sfapi_client_id, secret=state.sfapi_key
            ) as client:
                perlmutter = await client.compute(Machine.perlmutter)
                # set the target path where auxiliary files will be copied
                target_path = "/global/cfs/cdirs/m558/superfacility/model_training/src/"
                [target_path] = await perlmutter.ls(target_path, directory=True)
                # set the source path where auxiliary files are copied from
                source_path = Path.cwd().parent
                source_path_list = [
                    Path(source_path / "ml/train_model.py"),
                    Path(source_path / "ml/Neural_Net_Classes.py"),
                    Path(source_path / "dashboard/config/variables.yml"),
                ]
                # copy auxiliary files to NERSC
                for path in source_path_list:
                    with open(path, "rb") as f:
                        f.filename = path.name
                        await target_path.upload(f)
                # set the path of the script used to submit the training job on NERSC
                script_path = Path(source_path / "ml/training_pm.sbatch")
                with open(script_path, "r") as file:
                    script_job = file.read()
                # replace the --experiment command line argument in the batch script
                # with the current experiment in the state
                if state.model_type == "Neural Network":
                    script_job = re.sub(
                        pattern=r"--experiment (.*)",
                        repl=rf"--experiment {state.experiment} --model NN",
                        string=script_job,
                    )
                if state.model_type == "Gaussian Process":
                    script_job = re.sub(
                        pattern=r"--experiment (.*)",
                        repl=rf"--experiment {state.experiment} --model GP",
                        string=script_job,
                    )
                # submit the training job through the Superfacility API
                sfapi_job = await perlmutter.submit_job(script_job)
                state.model_training_status = "Submitted"
                state.flush()
                # print some logs
                print(f"Training job submitted (job ID: {sfapi_job.jobid})")
                return await monitor_sfapi_job(sfapi_job, "model_training_status")

        except Exception as e:
            title = "Unable to complete training kernel"
            msg = f"Error occurred when executing training kernel: {e}"
            add_error(title, msg)
            print(msg)

    async def training_async(self):
        try:
            print("Training model...")
            state.model_training = True
            state.model_training_status = "Submitting"
            state.flush()
            if await self.training_kernel():
                state.model_training_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                print(f"Finished training model at {state.model_training_time}")
            else:
                print("Unable to complete training job.")
            # flush state and enable button
            state.model_training = False
            state.flush()
        except Exception as e:
            title = "Unable to train model"
            msg = f"Error occurred when training model: {e}"
            add_error(title, msg)
            print(msg)

    def training_trigger(self):
        try:
            # schedule asynchronous job
            asyncio.create_task(self.training_async())
        except Exception as e:
            title = "Unable to train model"
            msg = f"Error occurred when training model: {e}"
            add_error(title, msg)
            print(msg)

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
                                disabled=(
                                    "model_training || perlmutter_status != 'active'",
                                ),
                                style="margin-left: 4px; margin-top: 12px; text-transform: none;",
                            )
                        with vuetify.VCol():
                            vuetify.VTextField(
                                v_model_number=("model_training_status",),
                                label="Training status",
                                readonly=True,
                                style="width: 100px;",
                            )
