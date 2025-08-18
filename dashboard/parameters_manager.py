import copy
import os
import yaml
import pandas as pd
from pathlib import Path
from sfapi_client import AsyncClient
from sfapi_client.compute import Machine
from calibration_manager import SimulationCalibrationManager
from trame.widgets import client, vuetify3 as vuetify
from state_manager import state
import asyncio
from utils import load_variables
from sfapi_manager import monitor_sfapi_job

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

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def reset(self):
        print("Resetting parameters to default values...")
        # reset parameters to initial values
        state.parameters = copy.deepcopy(state.parameters_init)
        # push again at flush time
        state.dirty("parameters")

    def optimize(self):
        print("Optimizing parameters...")
        # optimize parameters through model
        self.__model.optimize()

    async def simulate(self):
        setup = state.experiment
        print(f"\nExperiment parameters ({setup}):")
        print(state.parameters)

        input_variables, output_variables, simulation_calibration = load_variables()

        print(f"\nSimulation parameters ({setup}):")
        sim_data = {
            "var_name": [],
            "sim_val": []
             }

        getsim = SimulationCalibrationManager(simulation_calibration)
        sim_vals = getsim.convert_exp_to_sim(state.parameters, sim_data)
        print(sim_vals)

        save_dir = f"../simulation_data/{setup}"
        data_df = pd.DataFrame(sim_vals)
        data_df.to_csv(os.path.join(save_dir, "single_sim_vals.csv"), index=False)
        print("simulation values saved to csv")
        try:
            async with AsyncClient(
                client_id=state.sfapi_client_id, secret=state.sfapi_key
            ) as client:
                perlmutter = await client.compute(Machine.perlmutter)
                # set the target path where auxiliary files will be copied
                target_path1 = f"/global/cfs/cdirs/m558/superfacility/simulation_data/{setup}/"
                [target_path1] = await perlmutter.ls(target_path1, directory=True)
                target_path2 = f"/global/cfs/cdirs/m558/superfacility/simulation_data/{setup}/templates"
                [target_path2] = await perlmutter.ls(target_path2, directory=True)
                source_path = Path.cwd().parent
                source_path_list = [
                    Path(source_path / f"simulation_data/{setup}/templates/run_grid_scan.py"),
                    Path(source_path / f"simulation_data/{setup}/templates/warpx_input_script"),
                    Path(source_path / f"simulation_data/{setup}/single_sim_vals.csv")
                ]
                #copy auxiliary files to NERSC
                count = 1
                for path in source_path_list:
                    with open(path, "rb") as f:
                        f.filename = path.name
                        if count == 3:
                            await target_path1.upload(f)
                        else:
                            await target_path2.upload(f)
                        count += 1
                #set the path of the script used to submit the training job on NERSC
                script_path = Path(
                    source_path / f"simulation_data/{setup}/submission_script_single"
                )
                with open(script_path, "r") as file:
                    script_job = file.read()
                # submit the training job through the Superfacility API
                sfapi_job = await perlmutter.submit_job(script_job)
                if sfapi_job is None:
                    print("Error: Job submission failed")
                    state.simulation_job_status = "Submission Failed"
                    state.fluch()
                    return

                state.simulation_job_status = "Submitted"
                state.flush()
                # print some logs
                print(f"Simulation job submitted (job ID: {sfapi_job.jobid})")
                # wait for the job to move into a terminal state
                return await monitor_sfapi_job(sfapi_job, "simulation_job_status")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    def panel(self):
        print("Setting parameters card...")
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_parameters", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Parameters",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText():
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
                        # create a row for the buttons
                        with vuetify.VRow():
                            with vuetify.VCol():
                                vuetify.VBtn(
                                    "Reset",
                                    click=self.reset,
                                    style="margin-left: 4px; margin-top: 12px; text-transform: none;",
                                )
                            with vuetify.VCol():
                                vuetify.VBtn(
                                    "Optimize",
                                    click=self.optimize,
                                    style="margin-left: 12px; margin-top: 12px; text-transform: none;",
                                )

                        with vuetify.VRow():
                            with vuetify.VCol():
                                vuetify.VBtn(
                                    "Simulate",
                                    click=self.simulate,
                                    style="margin-right: 4px; margin-top: 12px; text-transform: none;",
                                )
                            with vuetify.VCol():
                                vuetify.VTextField(
                                    v_model_number=("simulation_job_status",),
                                    label="Simulation Status",
                                    readonly=True,
                                    style="width: 100px;",
                                )
