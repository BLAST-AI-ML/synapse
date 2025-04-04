from trame.widgets import vuetify2 as vuetify

from state_manager import state

class ParametersManager:

    def __init__(self, model, input_variables):
        # save PyTorch model
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

    def recenter(self):
        # recenter parameters
        for key in state.parameters.keys():
            state.parameters[key] = (state.parameters_min[key] + state.parameters_max[key]) / 2.
        # push again at flush time
        state.dirty("parameters")

    def optimize(self):
        # optimize parameters through model
        self.__model.optimize()

    def card(self):
        with vuetify.VCard():
            with vuetify.VCardTitle("Parameters"):
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
