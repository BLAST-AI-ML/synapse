import copy
from trame.widgets import vuetify2 as vuetify

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
        state.parameters_display_name = dict()
        for parameter_dict in input_variables.values():
            key = parameter_dict["name"]
            pmin = float(parameter_dict["value_range"][0])
            pmax = float(parameter_dict["value_range"][1])
            pval = float(parameter_dict["default"])
            state.parameters[key] = pval
            state.parameters_min[key] = pmin
            state.parameters_max[key] = pmax
            state.parameters_display_name[key] = parameter_dict.get("display_name", key)
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

    def panel(self):
        print("Setting parameters card...")
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_parameters", 0)):
            with vuetify.VExpansionPanel():
                vuetify.VExpansionPanelHeader(
                    "Control: Parameters", style="font-size: 20px; font-weight: 500;"
                )
                with vuetify.VExpansionPanelContent():
                    for count, key in enumerate(state.parameters.keys()):
                        pmin = state.parameters_min[key]
                        pmax = state.parameters_max[key]
                        step = (pmax - pmin) / 100.0
                        # create a row for the parameter label
                        with vuetify.VRow():
                            vuetify.VSubheader(
                                state.parameters_display_name[key],
                                style=(
                                    "margin-top: 16px;"
                                    if count == 0
                                    else "margin-top: 0px;"
                                ),
                            )
                        # create a row for the slider and text field
                        with vuetify.VRow(no_gutters=True):
                            with vuetify.VSlider(
                                v_model_number=(f"parameters['{key}']",),
                                change="flushState('parameters')",
                                hide_details=True,
                                max=pmax,
                                min=pmin,
                                step=step,
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
