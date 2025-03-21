from trame.widgets import vuetify2 as v2

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

    def update(self):
        for key in state.parameters.keys():
            state.parameters[key] = float(state.parameters[key])

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
        with v2.VCard():
            with v2.VCardTitle("Parameters"):
                #v2.VSpacer()
                ## create icon with tooltip to reset parameters
                #with v2.VTooltip(top=True):
                #    with v2.Template(v_slot_activator="{ on, attrs }"):
                #        with v2.VBtn(
                #            icon=True,
                #            click=self.recenter,
                #            target="_blank",
                #            v_bind="attrs",
                #            v_on="on",
                #        ):
                #            v2.VIcon("mdi-restart")
                #    v2.Template("Reset parameters")
                ## create icon with tooltip to optimize parameters
                #with v2.VTooltip(top=True):
                #    with v2.Template(v_slot_activator="{ on, attrs }"):
                #        with v2.VBtn(
                #            icon=True,
                #            click=self.optimize,
                #            target="_blank",
                #            v_bind="attrs",
                #            v_on="on",
                #        ):
                #            v2.VIcon("mdi-laptop")
                #    v2.Template("Optimize parameters")
                # create parameters sliders and text fields
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
                                        style="width: 80px;",
                                        type="number",
                                    )
