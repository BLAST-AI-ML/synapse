import asyncio
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import tempfile
import os
import yaml
import re
import mlflow
import mlflow.store.artifact.artifact_repo as mlflow_artifact_repo
import mlflow.store.artifact.cloud_artifact_repo as mlflow_cloud_artifact_repo
import mlflow.utils.file_utils as mlflow_file_utils
from mlflow.exceptions import MlflowException
from trame.assets.local import LocalFileManager
from sfapi_client import AsyncClient
from sfapi_client.compute import Machine
from trame.widgets import vuetify3 as vuetify, html
from utils import timer, load_config_dict, create_date_filter
from calibration_manager import build_inferred_calibration
from error_manager import add_error
from sfapi_manager import monitor_sfapi_job
from state_manager import state

LOGO_DIR = Path(__file__).parent / "logos"
AMSC_MLFLOW_URL = "https://mlflow.american-science-cloud.org"
GENESIS_MODEL_TYPE = "Neural Network (single)"
GENESIS_LOGO_PATH = LOGO_DIR / "genesis_80px.png"
AMSC_LOGO_PATH = LOGO_DIR / "AmSC_300px.png"
GENESIS_LOGO_URL = (
    LocalFileManager(LOGO_DIR).url("genesis_logo", GENESIS_LOGO_PATH)
    if GENESIS_LOGO_PATH.is_file()
    else None
)
AMSC_LOGO_URL = (
    LocalFileManager(LOGO_DIR).url("amsc_logo", AMSC_LOGO_PATH)
    if AMSC_LOGO_PATH.is_file()
    else None
)
MODEL_DOWNLOAD_ACTIVE_EXPR = (
    f"model_downloading && model_type_verbose === '{GENESIS_MODEL_TYPE}'"
)

model_type_dict = {
    "Gaussian Process": "GP",
    GENESIS_MODEL_TYPE: "NN",
    "Neural Network (ensemble)": "ensemble_NN",
}
AMSC_MLFLOW_MODEL_URL_EXPR = (
    f"'{AMSC_MLFLOW_URL}/#/models/synapse-' + experiment + '_' + "
    "(model_type_verbose === 'Gaussian Process' ? 'GP' : "
    f"model_type_verbose === '{GENESIS_MODEL_TYPE}' ? 'NN' : "
    "model_type_verbose === 'Neural Network (ensemble)' ? 'ensemble_NN' : "
    "model_type_verbose)"
)

_NO_PRELOADED_MODEL = object()


def clear_model_load_errors():
    """Remove stale model-load errors before starting another load attempt."""
    if state.errors is None:
        return
    errors = [
        error
        for error in state.errors
        if not str(error.get("title", "")).startswith("Unable to load model")
    ]
    if len(errors) != len(state.errors):
        state.errors = errors
        state.dirty("errors")


def is_missing_mlflow_model(error):
    """Return whether an MLflow error represents a missing registered model."""
    return isinstance(error, MlflowException) and (
        error.error_code == "RESOURCE_DOES_NOT_EXIST"
        or "RESOURCE_DOES_NOT_EXIST" in str(error)
    )


def load_model_from_mlflow(config_dict, model_type):
    """Load the latest registered MLflow model for an experiment configuration."""
    if "mlflow" not in config_dict or not config_dict["mlflow"].get("tracking_uri"):
        print(
            f"No mlflow.tracking_uri in configuration file for {config_dict['experiment']}; cannot load model from MLflow."
        )
        return None

    mlflow.set_tracking_uri(config_dict["mlflow"]["tracking_uri"])
    # When using the AmSC MLflow: inject the X-Api-Key into the requests to authenticate with the MLflow server
    # (See https://gitlab.com/amsc2/ai-services/model-services/intro-to-mlflow-pytorch)
    if config_dict["mlflow"]["tracking_uri"] == "https://mlflow.american-science-cloud.org":
        enable_amsc_x_api_key(config_dict)

    experiment = config_dict["experiment"]
    model_name = f"synapse-{experiment}_{model_type}"
    return (
        mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
        .unwrap_python_model()
        .model
    )


@contextmanager
def mlflow_artifact_progress_to_state(loop):
    """Expose MLflow artifact download progress through dashboard state."""
    progress_bar_modules = [
        mlflow_file_utils,
        mlflow_artifact_repo,
        mlflow_cloud_artifact_repo,
    ]
    original_progress_bars = {
        module: module.ArtifactProgressBar for module in progress_bar_modules
    }
    original_progress_bar = mlflow_file_utils.ArtifactProgressBar

    def set_download_progress(progress, total):
        """Publish the current download completion percentage to the GUI."""

        def update_progress_state():
            if total:
                state.model_download_progress = min(100, progress / total * 100)
            else:
                state.model_download_progress = None
            state.flush()

        loop.call_soon_threadsafe(update_progress_state)

    class TrameArtifactProgressBar(original_progress_bar):
        def __init__(self, desc, total, step, **kwargs):
            super().__init__(desc, total, step, **kwargs)
            self.trame_progress = 0
            if desc.startswith("Downloading"):
                set_download_progress(self.trame_progress, self.total)

        def update(self):
            super().update()
            self.trame_progress = min(
                self.total,
                self.trame_progress + self.step,
            )
            if self.desc.startswith("Downloading"):
                set_download_progress(self.trame_progress, self.total)

    for module in progress_bar_modules:
        module.ArtifactProgressBar = TrameArtifactProgressBar
    try:
        yield
    finally:
        for module, progress_bar in original_progress_bars.items():
            module.ArtifactProgressBar = progress_bar


def load_model_from_mlflow_with_progress(config_dict, model_type, loop):
    """Load an MLflow model while reporting artifact download progress."""
    with mlflow_artifact_progress_to_state(loop):
        return load_model_from_mlflow(config_dict, model_type)


def enable_amsc_x_api_key(config_dict):
    """
    MLflow authentication helper for the AmSC MLflow server.
    Standard MLflow does not automatically inject custom headers like 'X-Api-Key'.
    This patches the http_request function to ensure every request to the server
    includes the AmSC API key.

    See https://gitlab.com/amsc2/ai-services/model-services/intro-to-mlflow-pytorch for more details.
    """
    import mlflow.utils.rest_utils as rest_utils

    mlflow_cfg = config_dict.get("mlflow") or {}
    api_key_env = mlflow_cfg.get("api_key_env")
    if not api_key_env:
        title = "Unable to enable AmSC X-Api-Key authentication"
        msg = "MLFlow configuration is missing 'mlflow.api_key_env'"
        add_error(title, msg)
        print(msg)
        return

    api_key = os.environ.get(api_key_env)
    if not api_key:
        title = "Unable to enable AmSC X-Api-Key authentication"
        msg = f"Environment variable '{api_key_env}' in 'mlflow.api_key_env' is not set"
        add_error(title, msg)
        print(msg)
        return
    _orig = rest_utils.http_request

    def patched(host_creds, endpoint, method, *args, **kwargs):
        if "headers" in kwargs and kwargs["headers"] is not None:
            h = dict(kwargs["headers"])
            h["X-Api-Key"] = api_key
            kwargs["headers"] = h
        else:
            h = dict(kwargs.get("extra_headers") or {})
            h["X-Api-Key"] = api_key
            kwargs["extra_headers"] = h
        return _orig(host_creds, endpoint, method, *args, **kwargs)

    rest_utils.http_request = patched


class ModelManager:
    def __init__(self, config_dict, model_type, loaded_model=_NO_PRELOADED_MODEL):
        print("Initializing model manager...")
        clear_model_load_errors()
        self.__model = None
        self.__model_type = model_type

        experiment = config_dict["experiment"]
        model_name = f"synapse-{experiment}_{model_type}"

        try:
            self.__model = (
                load_model_from_mlflow(config_dict, model_type)
                if loaded_model is _NO_PRELOADED_MODEL
                else loaded_model
            )
            if self.__model is None:
                return
            if model_type not in ("NN", "ensemble_NN", "GP"):
                raise ValueError(f"Unsupported model type: {model_type}")
            # Populate inferred calibration in physics units for GUI
            # (only meaningful inside the dashboard where state.simulation_calibration is set)
            if state.simulation_calibration is not None:
                self.populate_inferred_calibration(
                    config_dict["inputs"], config_dict["outputs"]
                )
        except Exception as e:
            if is_missing_mlflow_model(e):
                print(f"Model {model_name} not found in MLflow; continuing without it.")
                return
            title = f"Unable to load model {model_type}"
            msg = f"Error occurred when loading model from MLflow: {e}"
            add_error(title, msg)
            print(msg)

    def avail(self):
        # print("Checking model availability...")
        model_avail = True if self.__model is not None else False
        return model_avail

    @timer
    def evaluate(self, parameters, output):
        print("Evaluating model...")
        if self.__model is not None:
            # evaluate model
            output_dict = self.__model.evaluate(parameters)
            if self.__model_type == "NN":
                # compute mean and mean error
                mean = output_dict[output]
                mean_error = 0.0  # trick to collapse error range when lower/upper bounds are not predicted
            elif self.__model_type in ("GP", "ensemble_NN"):
                # compute mean, standard deviation and mean error
                # (call detach method to detach gradients from tensors)
                mean = output_dict[output].mean.detach()
                std_dev = output_dict[output].variance.sqrt().detach()
                mean_error = 2.0 * std_dev
            else:
                raise ValueError(f"Unsupported model type: {self.__model_type}")
            # compute lower/upper bounds for error range
            lower = mean - mean_error
            upper = mean + mean_error
            # convert to Python float if tensor has only one element
            # because Trame state variables must be serializable
            if mean.numel() == 1:
                mean = float(mean)
            return (mean, lower, upper)

    def populate_inferred_calibration(self, input_variables, output_variables):
        """
        Populate alpha_inferred/beta_inferred in state.simulation_calibration for
        both input and output calibration entries.
        """
        # Clear stale inferred values
        for value in state.simulation_calibration.values():
            value.pop("alpha_inferred", None)
            value.pop("beta_inferred", None)

        # Input calibration
        # For ensemble_NN, transformers live on each inner TorchModel (not on NNEnsemble itself)
        if self.__model_type == "ensemble_NN":
            input_transformers = self.__model.models[0].input_transformers
        else:
            input_transformers = self.__model.input_transformers
        assert len(input_transformers) == 2, (
            f"Expected exactly 2 input transformers (calibration + normalization), "
            f"but got {len(input_transformers)}."
        )
        input_inferred_calibration = input_transformers[0]
        alpha_inferred = 1.0 / input_inferred_calibration.coefficient
        beta_inferred = input_inferred_calibration.offset
        build_inferred_calibration(input_variables, alpha_inferred, beta_inferred)

        # Output calibration
        # For ensemble_NN, transformers live on each inner TorchModel (not on NNEnsemble itself)
        if self.__model_type == "ensemble_NN":
            output_transformers = self.__model.models[0].output_transformers
        else:
            output_transformers = self.__model.output_transformers
        assert len(output_transformers) == 2, (
            f"Expected exactly 2 output transformers (normalization + calibration), "
            f"but got {len(output_transformers)}."
        )
        output_inferred_calibration = output_transformers[-1]
        alpha_inferred = 1.0 / output_inferred_calibration.coefficient
        beta_inferred = output_inferred_calibration.offset
        build_inferred_calibration(output_variables, alpha_inferred, beta_inferred)
        # Notify Trame that the dict was modified in-place, so the UI updates
        state.dirty("simulation_calibration")

    def _prepare_training_config(self, temp_dir):
        """Prepare a training configuration file in the given temporary directory,
        updated with information from the dashboard.

        Returns the path to the written configuration file.
        """
        config_dict = load_config_dict(state.experiment)
        config_dict["simulation_calibration"] = state.simulation_calibration
        date_filter = create_date_filter(state.experiment_date_range)
        config_dict["date_filter"] = date_filter
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)
        return config_path

    async def _training_kernel_sfapi(self):
        try:
            # create an authenticated client
            async with AsyncClient(
                client_id=state.sfapi_client_id, secret=state.sfapi_key
            ) as client:
                perlmutter = await client.compute(Machine.perlmutter)
                # upload the configuration file to NERSC
                with tempfile.TemporaryDirectory() as temp_dir:
                    config_path = self._prepare_training_config(temp_dir)
                    # define the target path on NERSC
                    target_path = "/global/cfs/cdirs/m558/superfacility/model_training"
                    [target_path] = await perlmutter.ls(target_path, directory=True)
                    with open(config_path, "rb") as temp_file:
                        print("Uploading configuration file to NERSC...")
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
                    repl=rf"--model {model_type_dict[state.model_type_verbose]}",
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
            title = "Unable to complete remote training"
            msg = f"Error occurred when executing remote training: {e}"
            add_error(title, msg)
            print(msg)
            state.model_training_status = "Failed"
            state.flush()
            return False

    async def _training_kernel_local(self):
        try:
            ml_dir = (Path(__file__).parent / "../ml").resolve()
            train_model_path = ml_dir / "train_model.py"
            model_type = model_type_dict[state.model_type_verbose]

            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = self._prepare_training_config(temp_dir)
                state.model_training_status = "Running"
                state.flush()
                print(
                    f"Starting local training: {train_model_path} --model {model_type}"
                )
                proc = await asyncio.create_subprocess_exec(
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    "synapse-ml",
                    "python",
                    str(train_model_path),
                    "--config_file",
                    str(config_path),
                    "--model",
                    model_type,
                    cwd=str(ml_dir),  # working directory of the subprocess
                    stdout=asyncio.subprocess.PIPE,  # capture the standard output into a pipe
                    stderr=asyncio.subprocess.STDOUT,  # redirect the standard error into the same pipe
                )
                # stream subprocess output to console
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    print(line.decode(), end="")
                await proc.wait()

            if proc.returncode == 0:
                state.model_training_status = "Completed"
                state.flush()
                return True
            else:
                state.model_training_status = "Failed"
                state.flush()
                return False
        except Exception as e:
            title = "Unable to complete local training"
            msg = f"Error occurred when executing local training: {e}"
            add_error(title, msg)
            print(msg)
            state.model_training_status = "Failed"
            state.flush()
            return False

    async def training_async(self):
        try:
            print("Training model...")
            state.model_training = True
            state.model_training_status = "Submitting"
            state.flush()
            if state.model_training_mode == "local":
                result = await self._training_kernel_local()
            elif state.model_training_mode == "sfapi":
                result = await self._training_kernel_sfapi()
            else:
                raise ValueError(
                    f"Unsupported training mode: {state.model_training_mode}"
                )
            if result:
                state.model_training_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                state.flush()
                print(f"Finished training model at {state.model_training_time}")
            else:
                print("Unable to complete training job")
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
            {"title": "Gaussian Process", "value": "Gaussian Process"},
            {"title": GENESIS_MODEL_TYPE, "value": GENESIS_MODEL_TYPE},
            {
                "title": "Neural Network (ensemble)",
                "value": "Neural Network (ensemble)",
            },
        ]
        if GENESIS_LOGO_URL:
            model_type_list[1]["logo"] = GENESIS_LOGO_URL
        model_type_cols = 8 if AMSC_LOGO_URL else 12
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_model", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Models",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText():
                    with vuetify.VRow(align="center"):
                        with vuetify.VCol(cols=model_type_cols):
                            with vuetify.VSelect(
                                v_model=("model_type_verbose",),
                                label="Model type",
                                items=(model_type_list,),
                                item_title="title",
                                item_value="value",
                                dense=True,
                            ):
                                with html.Template(v_slot_item="{ props, item }"):
                                    with vuetify.VListItem(
                                        v_bind=("{ ...props, title: undefined }",)
                                    ):
                                        with html.Div(
                                            classes="d-flex align-center w-100"
                                        ):
                                            html.Span(v_text=("item.title",))
                                            vuetify.VSpacer()
                                            vuetify.VImg(
                                                v_if=("item.raw.logo",),
                                                src=("item.raw.logo",),
                                                width=40,
                                                height=24,
                                                max_width=40,
                                                alt="Genesis",
                                                style="margin-left: 12px;",
                                            )
                        if AMSC_LOGO_URL:
                            with vuetify.VCol(
                                cols=4,
                                classes="d-flex align-center justify-end",
                            ):
                                with html.A(
                                    v_if=("model_available",),
                                    href=(AMSC_MLFLOW_MODEL_URL_EXPR,),
                                    target="_blank",
                                    rel="noopener noreferrer",
                                    title="Open selected model in AmSC MLflow",
                                    style=(
                                        "display: block; width: 100%; "
                                        "max-width: 300px; margin-left: auto; "
                                        "cursor: pointer;"
                                    ),
                                ):
                                    vuetify.VImg(
                                        src=AMSC_LOGO_URL,
                                        alt="AmSC",
                                        max_width=300,
                                        max_height=72,
                                        contain=True,
                                        style="width: 100%;",
                                    )
                                vuetify.VImg(
                                    v_if=("!model_available",),
                                    src=AMSC_LOGO_URL,
                                    alt="AmSC",
                                    max_width=300,
                                    max_height=72,
                                    contain=True,
                                    title="Selected model is not available in AmSC MLflow",
                                    style=(
                                        "width: 100%; max-width: 300px; "
                                        "margin-left: auto;"
                                    ),
                                )
                    with vuetify.VRow(
                        v_if=(MODEL_DOWNLOAD_ACTIVE_EXPR,),
                        no_gutters=True,
                        align="center",
                        style="margin-top: -8px; margin-bottom: 8px;",
                    ):
                        with vuetify.VCol(cols=model_type_cols):
                            with html.Div(
                                classes="d-flex align-center text-caption text-medium-emphasis mb-1"
                            ):
                                vuetify.VIcon(
                                    "mdi-cloud-download-outline",
                                    size=16,
                                    classes="mr-1",
                                )
                                html.Span(v_text=("model_download_status",))
                                vuetify.VSpacer()
                                html.Span(
                                    v_if=("model_download_progress !== null",),
                                    v_text=("`${Math.round(model_download_progress)}%`",),
                                )
                            vuetify.VProgressLinear(
                                indeterminate=("model_download_progress === null",),
                                model_value=("model_download_progress",),
                                color="primary",
                                height=4,
                                rounded=True,
                            )
                    with vuetify.VRow(align="center"):
                        with vuetify.VCol(cols="auto"):
                            vuetify.VBtn(
                                "Train",
                                click=self.training_trigger,
                                disabled=(
                                    "model_training || (model_training_mode === 'sfapi' && perlmutter_status !== 'active')",
                                ),
                                style="text-transform: none",
                            )
                        with vuetify.VCol(cols=6, style="margin-left: auto;"):
                            vuetify.VTextField(
                                v_model_number=("model_training_status",),
                                label="Training status",
                                readonly=True,
                            )
