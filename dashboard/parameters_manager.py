import copy
from trame.widgets import vuetify2 as vuetify
import os
import yaml
from state_manager import state


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

    def simulate(self):
        setup = state.experiment
        print(f"\nExperimental Values {setup}")
        for name in state.parameters:
            print(f'{name}: {state.parameters[name]}')

        # find configuration file in the local file system
        config_dir = os.path.join(os.getcwd(), "config")
        config_file = os.path.join(config_dir, "variables.yml")
        if not os.path.isfile(config_file):
            raise ValueError(f"Configuration file {config_file} not found")
        
        print("Loading configuration dictionary...")
        # read configuration file
        with open(config_file) as f:
            config_str = f.read()
        # load configuration dictionary
        config_dict = yaml.safe_load(config_str)
        
        # load configuration dictionary
        config_spec = config_dict[state.experiment]
        # dictionary of input variables (parameters)
        input_variables = config_spec["input_variables"]
        # dictionary of output variables (objectives)
        # dictionary of calibration variables
        simulation_calibration = config_spec["simulation_calibration"]

        print(f"\nSimulation Values {setup}")
        for name in state.parameters:
            for sim_name, sim_info in simulation_calibration.items():
                if sim_info["depends_on"] == name:
                    alpha = sim_info["alpha"]
                    beta = sim_info["beta"]
                    sim_val = alpha * (state.parameters[name] - beta)
                    print(f"{sim_info['name']}: {sim_val}")

            
        
    def panel(self):
        print("Setting parameters card...")
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_parameters", 0)):
            with vuetify.VExpansionPanel():
                vuetify.VExpansionPanelHeader(
                    "Control: Parameters",
                    style="font-size: 20px; font-weight: 500;",
                )
                with vuetify.VExpansionPanelContent():
                    for count, key in enumerate(state.parameters.keys()):
                        # create a row for the parameter label
                        with vuetify.VRow():
                            vuetify.VSubheader(
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
                                        style="margin-top: 0px; padding-top: 0px; width: 80px;",
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
                                    style="width: 80px;",
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
                                    style="width: 80px;",
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
                        with vuetify.VCol():
                            vuetify.VBtn(
                                "Simulation",
                                click=self.simulate,
                                style="margin-left: 12px; margin-top: 12px; text-transform: none;",
                            )
