import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import os
import yaml
import re
from sfapi_client import AsyncClient
from sfapi_client.compute import Machine
from lume_model.models.torch_model import TorchModel
from lume_model.models.ensemble import NNEnsemble
from lume_model.models.gp_model import GPModel
from trame.widgets import vuetify3 as vuetify
from utils import verify_input_variables, timer, load_config_dict
from error_manager import add_error
from sfapi_manager import monitor_sfapi_job
from state_manager import state

model_type_tag_dict = {
    "Gaussian Process": "GP",
    "Neural Network (single)": "NN",
    "Neural Network (ensemble)": "ensemble_NN",
}


class ModelManager:
    def __init__(self, db):
        print("Initializing model manager...")
        # Set initial default values
        self.__model = None
        self.__is_neural_network = False
        self.__is_gaussian_process = False
        self.__is_neural_network_ensemble = False

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
            # Open content of the top-level YAML file
            yaml_file_content = document["yaml_file_content"]
            model_filename = f"{state.experiment}.yml"
            with open(os.path.join(temp_dir, model_filename), "w") as f:
                f.write(yaml_file_content)

            # Extract list of files to download
            files_to_download = []
            if state.model_type == "Neural Network (ensemble)":
                models_info = yaml.safe_load(yaml_file_content)
                # Download yaml file for each model within the ensemble
                for model in models_info["models"]:
                    yaml_file_name = model.replace("_model.jit", ".yml")
                    with open(os.path.join(temp_dir, yaml_file_name), "wb") as f:
                        f.write(document[yaml_file_name])
                    model_info = yaml.safe_load(document[yaml_file_name])
                    # Extract files to download
                    files_to_download += (
                        [model_info["model"]]
                        + model_info["input_transformers"]
                        + model_info["output_transformers"]
                    )
            else:
                # Extract files to download
                model_info = yaml.safe_load(yaml_file_content)
                files_to_download = (
                    [model_info["model"]]
                    + model_info["input_transformers"]
                    + model_info["output_transformers"]
                )

            # Download all the files that define the model(s)
            for filename in files_to_download:
                with open(os.path.join(temp_dir, filename), "wb") as f:
                    f.write(document[filename])

            # Check consistency of the model file
            print("Reading model file...")
            model_file = os.path.join(temp_dir, f"{state.experiment}.yml")
            if not os.path.isfile(model_file):
                title = f"Model file {model_file} not found"
                msg = f"Unable to find the model file for {state.experiment}"
                add_error(title, msg)
                print(msg)
                return
            elif not verify_input_variables(model_file, state.experiment):
                title = "Model file input variable mismatch"
                msg = f"Model file {model_file} has different input variables than the configuration file for {state.experiment}"
                add_error(title, msg)
                print(msg)
                return

            # Load model with lume_model
            try:
                if state.model_type == "Neural Network (single)":
                    self.__is_neural_network = True
                    self.__model = TorchModel(model_file)
                elif state.model_type == "Neural Network (ensemble)":
                    self.__is_neural_network_ensemble = True
                    self.__model = NNEnsemble(model_file)
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

    @property
    def is_neural_network_ensemble(self):
        return self.__is_neural_network_ensemble

    @timer
    def evaluate(self, parameters, output):
        print("Evaluating model...")
        if self.__model is not None:
            # evaluate model
            output_dict = self.__model.evaluate(parameters)
            if self.__is_neural_network:
                # compute mean and mean error
                mean = output_dict[output]
                mean_error = 0.0  # trick to collapse error range when lower/upper bounds are not predicted
            elif self.__is_gaussian_process or self.__is_neural_network_ensemble:
                if self.__is_gaussian_process:
                    # TODO use "exp" only once experimental data is available for all experiments
                    task_tag = "exp" if state.experiment == "bella-ip2" else "sim"
                    output_key = [key for key in output_dict.keys() if task_tag in key][
                        0
                    ]
                elif self.__is_neural_network_ensemble:
                    output_key = list(output_dict.keys())[0]

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
                # Upload the config.yaml to nersc
                config_dict = load_config_dict(state.experiment)
                config_dict["simulation_calibration"] = state.simulation_calibration
                target_path = "/global/cfs/cdirs/m558/superfacility/model_training"
                [target_path] = await perlmutter.ls(target_path, directory=True)
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = Path(temp_dir) / "config.yaml"
                    with open(temp_file_path, "w") as temp_file:
                        yaml.dump(config_dict, temp_file)
                        temp_file.flush()
                    with open(temp_file_path, "rb") as temp_file:
                        print("Uploading config file to NERSC")
                        temp_file.filename = "config.yaml"
                        await target_path.upload(temp_file)

                # set the path of the script used to submit the training job on NERSC
                training_script = None
                # multiple locations supported, to make development easier
                #   container (production): script is in cwd
                #   development, starting the gui app from dashboard/: script is in ../ml/
                #   development, starting the gui app from the repo root dir: script is in ml/
                script_locations = [Path.cwd(), Path.cwd() / "../ml", Path.cwd() / "ml"]
                for script_dir in script_locations:
                    script_path = script_dir / "training_pm.sbatch"
                    if os.path.exists(script_path):
                        with open(script_path, "r") as file:
                            training_script = file.read()
                        break
                if training_script is None:
                    raise RuntimeError("Could not find training_pm.sbatch")

                # replace the --model argument in the python command with the current model type from the state
                training_script = re.sub(
                    pattern=r"--model \$\{model\}",
                    repl=rf"--model {model_type_tag_dict[state.model_type]}",
                    string=training_script,
                )

                # submit the training job through the Superfacility API
                sfapi_job = await perlmutter.submit_job(training_script)
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
                state.flush()
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
            "Neural Network (single)",
            "Neural Network (ensemble)",
        ]
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_model", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Models",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText():
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VSelect(
                                v_model=("model_type",),
                                label="Model type",
                                items=(model_type_list,),
                                dense=True,
                            )
                        with vuetify.VCol():
                            vuetify.VTextField(
                                v_model_number=("model_training_status",),
                                label="Training status",
                                readonly=True,
                            )
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VBtn(
                                "Train",
                                click=self.training_trigger,
                                disabled=(
                                    "model_training || perlmutter_status != 'active'",
                                ),
                                style="text-transform: none",
                            )
