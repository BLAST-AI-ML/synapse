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
        for _, parameter_dict in input_variables.items():
            key = parameter_dict["name"]
            pmin = float(parameter_dict["value_range"][0])
            pmax = float(parameter_dict["value_range"][1])
            pval = float(parameter_dict["default"])
            state.parameters[key] = pval
            state.parameters_min[key] = pmin
            state.parameters_max[key] = pmax
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

    def card(self):
        print("Setting parameters card...")
        with vuetify.VCard():
            with vuetify.VCardTitle("Control: Parameters"):
                vuetify.VSpacer()
                with vuetify.VTooltip(bottom=True):
                    with vuetify.Template(v_slot_activator="{ on, attrs }"):
                        with vuetify.VBtn(
                            icon=True,
                            click=self.optimize,
                            v_on="on",
                            v_bind="attrs",
                        ):
                            vuetify.VIcon("mdi-calculator-variant")
                    vuetify.Template("Optimize")
                with vuetify.VTooltip(bottom=True):
                    with vuetify.Template(v_slot_activator="{ on, attrs }"):
                        with vuetify.VBtn(
                            icon=True,
                            click=self.reset,
                            v_on="on",
                            v_bind="attrs",
                        ):
                            vuetify.VIcon("mdi-restart")
                    vuetify.Template("Reset")
                with vuetify.VCardText():
                    for key in state.parameters.keys():
                        pmin = state.parameters_min[key]
                        pmax = state.parameters_max[key]
                        step = (pmax - pmin) / 100.
                        # create a row for the parameter label
                        with vuetify.VRow():
                            vuetify.VSubheader(key)
                        # create a row for the slider and text field
                        with vuetify.VRow(no_gutters=True):
                            with vuetify.VSlider(
                                v_model_number=(f"parameters['{key}']",),
                                change="flushState('parameters')",
                                classes="align-center",
                                hide_details=True,
                                max=pmax,
                                min=pmin,
                                step=step,
                            ):
                                with vuetify.Template(v_slot_append=True):
                                    vuetify.VTextField(
                                        v_model_number=(f"parameters['{key}']",),
                                        classes="mt-0 pt-0",
                                        density="compact",
                                        hide_details=True,
                                        readonly=True,
                                        single_line=True,
                                        style="width: 80px;",
                                        type="number",
                                    )
