from trame.widgets import vuetify2 as v2

from state_manager import state

class ParametersManager:

    def __init__(self, server, input_variables):
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

    def get(self):
        return state.parameters

    def get_min(self):
        return state.parameters_min

    def get_max(self):
        return state.parameters_max

    def update(self):
        for key in state.parameters.keys():
            state.parameters[key] = float(state.parameters[key])

    def recenter(self):
        # recenter parameters
        for key in state.parameters.keys():
            state.parameters[key] = (state.parameters_min[key] + state.parameters_max[key]) / 2.
        # push again at flush time
        state.dirty("parameters")

    def card(self):
        with v2.VCard():
            with v2.VCardTitle("Parameters"):
                with v2.VCardText():
                    for key in state.parameters.keys():
                        pmin = state.parameters_min[key]
                        pmax = state.parameters_max[key]
                        step = (pmax - pmin) / 100.
                        # create a row for the parameter label
                        with v2.VRow():
                            v2.VSubheader(key)
                        # create a row for the slider and text field
                        with v2.VRow(no_gutters=True):
                            with v2.VSlider(
                                v_model_number=(f"parameters['{key}']",),
                                change="flushState('parameters')",
                                classes="align-center",
                                hide_details=True,
                                max=pmax,
                                min=pmin,
                                step=step,
                            ):
                                with v2.Template(v_slot_append=True):
                                    v2.VTextField(
                                        v_model_number=(f"parameters['{key}']",),
                                        classes="mt-0 pt-0",
                                        density="compact",
                                        hide_details=True,
                                        readonly=True,
                                        single_line=True,
                                        style="width: 80px",
                                        type="number",
                                    )
