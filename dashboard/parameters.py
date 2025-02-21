from trame.widgets import vuetify2 as v2

class Parameters:

    def __init__(self, server, input_variables):
        # Trame state and controller
        self.__state = server.state
        self.__ctrl = server.controller
        # define state variables
        self.__state.parameters = dict()
        self.__state.parameters_min = dict()
        self.__state.parameters_max = dict()
        for _, parameter_dict in input_variables.items():
            key = parameter_dict["name"]
            pmin = float(parameter_dict["value_range"][0])
            pmax = float(parameter_dict["value_range"][1])
            pval = float(parameter_dict["default"])
            self.__state.parameters[key] = pval
            self.__state.parameters_min[key] = pmin
            self.__state.parameters_max[key] = pmax
        ## TODO decorate method 'recenter'?
        #self.recenter = self.__ctrl.add("recenter")(self.recenter)

    def get(self):
        return self.__state.parameters

    def get_min(self):
        return self.__state.parameters_min

    def get_max(self):
        return self.__state.parameters_max

    def update(self):
        for key in self.__state.parameters.keys():
            self.__state.parameters[key] = float(self.__state.parameters[key])

    def recenter(self):
        # recenter parameters
        for key in self.__state.parameters.keys():
            self.__state.parameters[key] = (self.__state.parameters_min[key] + self.__state.parameters_max[key]) / 2.
        # push again at flush time
        self.__state.dirty("parameters")

    def card(self):
        with v2.VCard(style="width: 500px"):
            with v2.VCardTitle("Parameters"):
                with v2.VCardText():
                    for key in self.__state.parameters.keys():
                        pmin = self.__state.parameters_min[key]
                        pmax = self.__state.parameters_max[key]
                        step = (pmax - pmin) / 100.
                        # create slider for each parameter
                        with v2.VSlider(
                            v_model_number=(f"parameters['{key}']",),
                            change="flushState('parameters')",
                            label=key,
                            min=pmin,
                            max=pmax,
                            step=step,
                            classes="align-center",
                            hide_details=True,
                            type="number",
                        ):
                            # append text field
                            with v2.Template(v_slot_append=True):
                                v2.VTextField(
                                    v_model_number=(f"parameters['{key}']",),
                                    label=key,
                                    density="compact",
                                    hide_details=True,
                                    readonly=True,
                                    single_line=True,
                                    style="width: 100px",
                                    type="number",
                                )

