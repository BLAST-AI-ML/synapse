from trame.widgets import vuetify2 as v2

class Objectives:
    def __init__(self, server, model, output_variables):
        # FIXME generalize for multiple objectives
        assert len(output_variables) == 1, "number of objectives > 1 not supported"
        # Trame state and controller
        self.__state = server.state
        self.__ctrl = server.controller
        # save PyTorch model
        self.__model = model
        # define state variables
        self.__state.objectives = dict()
        for _, objective_dict in output_variables.items():
            key = objective_dict["name"]
            if model.avail():
                self.__state.objectives[key] = model.evaluate(self.__state.parameters)
            else:
                print(f"Objectives.__init__: Model not provided, skip initialization")
                print(f"Objectives.__init__: Could not compute state.objectives[{key}]")
                self.__state.objectives[key] = None

    def get(self):
        return self.__state.objectives

    def update(self):
        for key in self.__state.objectives.keys():
            if self.__model.avail():
                self.__state.objectives[key] = self.__model.evaluate(self.__state.parameters)
            else:
                print(f"Objectives.update: Model not provided, skip update")
        # push again at flush time
        self.__state.dirty("objectives")

    def card(self):
        with v2.VCard():
            with v2.VCardTitle("Objectives"):
                with v2.VCardText():
                    for key in self.__state.objectives.keys():
                        # create a row for the objective label
                        with v2.VRow():
                            v2.VSubheader(key)
                        # create a row for the text field
                        with v2.VRow(no_gutters=True):
                            v2.VTextField(
                                v_model_number=(f"objectives['{key}']",),
                                classes="mt-0 pt-0",
                                density="compact",
                                hide_details=True,
                                readonly=True,
                                single_line=True,
                                type="number",
                            )
