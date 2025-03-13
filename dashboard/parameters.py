from trame.widgets import client, vuetify3 as vuetify

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
        with vuetify.VCard():
            with vuetify.VCardTitle("Parameters"):
                with vuetify.VCardText():
                    with client.DeepReactive("parameters") as dr:
                        for key in self.__state.parameters.keys():
                            pmin = self.__state.parameters_min[key]
                            pmax = self.__state.parameters_max[key]
                            step = (pmax - pmin) / 100.
                            # create a row for the text field
                            with vuetify.VRow():
                                vuetify.VTextField(
                                    v_model_number=(f"parameters['{key}']",),
                                    label=key,
                                    readonly=True,
                                    type="number",
                                    classes="mt-2 mb-0",
                                )
                            # create a row for the slider
                            with vuetify.VRow(no_gutters=True, classes="py-0"):
                                vuetify.VSlider(
                                    v_model_number=(f"parameters['{key}']",),
                                    min=pmin,
                                    max=pmax,
                                    step=step,
                                    classes="mt-0",
                                    hide_details=True,
                                    type="number",
                                )
