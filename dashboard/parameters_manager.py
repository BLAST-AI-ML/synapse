import asyncio
import copy
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from sfapi_client import AsyncClient
from sfapi_client.compute import Machine
from trame.widgets import client, vuetify3 as vuetify
from utils import load_variables
from calibration_manager import SimulationCalibrationManager
from error_manager import add_error
from sfapi_manager import monitor_sfapi_job
from state_manager import state, EXPERIMENTS_PATH


class ParametersManager:
    def __init__(self, model, input_variables):
        print("Initializing parameters manager...")
        # save model
        self.__model = model
        # define state variables
        state.parameters = dict()
        state.parameters_min = dict()
        state.parameters_max = dict()
        state.parameters_show_all = dict()
        self.parameters_step = dict()
        state.simulatable = (
            self.simulation_scripts_base_path / "submission_script_single"
        ).is_file()
        for _, parameter_dict in input_variables.items():
            key = parameter_dict["name"]
            pmin = float(parameter_dict["value_range"][0])
            pmax = float(parameter_dict["value_range"][1])
            pval = float(parameter_dict["default"])
            state.parameters[key] = pval
            state.parameters_min[key] = pmin
            state.parameters_max[key] = pmax
            state.parameters_show_all[key] = False
            self.parameters_step[key] = (pmax - pmin) / 100
        state.parameters_init = copy.deepcopy(state.parameters)
        # define default dislpayed inputs
        state.displayed_inputs = "Experiment"

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    @property
    def simulation_scripts_base_path(self):
        return EXPERIMENTS_PATH / f"synapse-{state.experiment}/simulation_scripts/"

    def reset(self):
        print("Resetting parameters to default values...")
        # reset parameters to initial values
        state.parameters = copy.deepcopy(state.parameters_init)
        # push again at flush time
        state.dirty("parameters")

    async def simulation_kernel(self):
        try:
            # create an authenticated client
            async with AsyncClient(
                client_id=state.sfapi_client_id, secret=state.sfapi_key
            ) as client:
                perlmutter = await client.compute(Machine.perlmutter)
                # set the target path where auxiliary files will be copied
                target_path = f"/global/cfs/cdirs/m558/superfacility/simulation_running/{state.experiment}/templates"
                [target_path] = await perlmutter.ls(target_path, directory=True)
                # set the base path where auxiliary files are copied from
                with tempfile.TemporaryDirectory() as temp_dir:
                    # store the current simulation parameters in a YAML temporary file
                    temp_file_path = (
                        Path(temp_dir) / "single_simulation_parameters.yaml"
                    )
                    _, _, simulation_calibration = load_variables(state.experiment)
                    sim_cal = SimulationCalibrationManager(simulation_calibration)
                    sim_dict = sim_cal.convert_exp_to_sim(state.parameters)
                    with open(temp_file_path, "w") as temp_file:
                        yaml.dump(sim_dict, temp_file)
                        temp_file.flush()
                    # set the source path where auxiliary files are copied from
                    source_paths = [
                        file
                        for file in (
                            self.simulation_scripts_base_path / "templates/"
                        ).rglob("*")
                        if file.is_file()
                    ] + [temp_file_path]
                    # copy auxiliary files to NERSC
                    for path in source_paths:
                        print(f"Uploading file to NERSC: {path}")
                        with open(path, "rb") as f:
                            f.filename = path.name
                            await target_path.upload(f)
                # set the path of the script used to submit the simulation job on NERSC
                with open(
                    self.simulation_scripts_base_path / "submission_script_single", "r"
                ) as file:
                    submission_script = file.read()
                # submit the simulation job through the Superfacility API
                print("Submitting job to NERSC")
                sfapi_job = await perlmutter.submit_job(submission_script)
                state.simulation_running_status = "Submitted"
                state.flush()
                # print some logs
                print(f"Simulation job submitted (job ID: {sfapi_job.jobid})")
                return await monitor_sfapi_job(sfapi_job, "simulation_running_status")
        except Exception as e:
            title = "Unable to complete simulation kernel"
            msg = f"Error occurred when executing simulation kernel: {e}"
            add_error(title, msg)
            print(msg)
            state.simulation_running_status = "Failed"

    async def simulation_async(self):
        try:
            print("Running simulation...")
            state.simulation_running = True
            state.simulation_running_status = "Submitting"
            state.flush()
            if await self.simulation_kernel():
                state.simulation_running_time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                )
                print(f"Finished running simulation at {state.simulation_running_time}")
            else:
                print("Unable to complete simulation job.")
            # flush state and enable button
            state.simulation_running = False
            state.flush()
        except Exception as e:
            title = "Unable to run simulation"
            msg = f"Error occurred when running simulation: {e}"
            add_error(title, msg)
            print(msg)

    def simulation_trigger(self):
        try:
            # schedule asynchronous job
            asyncio.create_task(self.simulation_async())
        except Exception as e:
            title = "Unable to run simulation"
            msg = f"Error occurred when running simulation: {e}"
            add_error(title, msg)
            print(msg)

    def panel(self):
        print("Setting parameters card...")
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_parameters", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Parameters",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText():
                    with vuetify.VRow():
                        vuetify.VSelect(
                            v_model=("displayed_output",),
                            items=(state.output_variables,),
                            dense=True,
                            label="Displayed output",
                        )
                    with vuetify.VRow():
                        vuetify.VSelect(
                            v_model=("displayed_inputs",),
                            items=(["Experiment", "Simulation"],),
                            dense=True,
                            label="Displayed inputs",
                        )
                    with client.DeepReactive("parameters"):
                        for count, key in enumerate(state.parameters.keys()):
                            # create a row for the parameter label
                            with vuetify.VRow():
                                vuetify.VListSubheader(
                                    key,
                                    style=(
                                        "margin-top: 16px;"
                                        if count == 0
                                        else "margin-top: 0px;"
                                    ),
                                )
                            with vuetify.VRow(no_gutters=True):
                                with vuetify.VSlider(
                                    v_model_number=(f"parameters['{key}']",),
                                    change="flushState('parameters')",
                                    hide_details=True,
                                    min=(f"parameters_min['{key}']",),
                                    max=(f"parameters_max['{key}']",),
                                    step=(
                                        f"(parameters_max['{key}'] - parameters_min['{key}']) / 100",
                                    ),
                                    style="align-items: center;",
                                ):
                                    with vuetify.Template(v_slot_append=True):
                                        vuetify.VTextField(
                                            v_model_number=(f"parameters['{key}']",),
                                            density="compact",
                                            hide_details=True,
                                            readonly=True,
                                            single_line=True,
                                            style="margin-top: 0px; padding-top: 0px; width: 100px;",
                                            type="number",
                                        )
                            step = self.parameters_step[key]
                            with vuetify.VRow(no_gutters=True):
                                with vuetify.VCol():
                                    vuetify.VTextField(
                                        v_model_number=(f"parameters_min['{key}']",),
                                        change="flushState('parameters_min')",
                                        density="compact",
                                        hide_details=True,
                                        disabled=(f"parameters_show_all['{key}']",),
                                        step=step,
                                        __properties=["step"],
                                        style="width: 100px;",
                                        type="number",
                                        label="min",
                                    )
                                with vuetify.VCol():
                                    vuetify.VTextField(
                                        v_model_number=(f"parameters_max['{key}']",),
                                        change="flushState('parameters_max')",
                                        density="compact",
                                        hide_details=True,
                                        disabled=(f"parameters_show_all['{key}']",),
                                        step=step,
                                        __properties=["step"],
                                        style="width: 100px;",
                                        type="number",
                                        label="max",
                                    )
                                with vuetify.VCol(style="min-width: 100px;"):
                                    vuetify.VCheckbox(
                                        v_model=(
                                            f"parameters_show_all['{key}']",
                                            False,
                                        ),
                                        density="compact",
                                        change="flushState('parameters_show_all')",
                                        label="Show all",
                                    )
                        with vuetify.VRow(align="center"):
                            with vuetify.VCol(cols=6):
                                with vuetify.VRow():
                                    with vuetify.VCol():
                                        vuetify.VBtn(
                                            "Reset",
                                            click=self.reset,
                                            style="text-transform: none",
                                        )
                                    with vuetify.VCol():
                                        vuetify.VBtn(
                                            "Simulate",
                                            click=self.simulation_trigger,
                                            disabled=(
                                                "simulation_running || perlmutter_status != 'active' || !simulatable",
                                            ),
                                            style="text-transform: none;",
                                        )
                            with vuetify.VCol(cols=6):
                                vuetify.VTextField(
                                    v_model_number=("simulation_running_status",),
                                    label="Simulation status",
                                    readonly=True,
                                )
