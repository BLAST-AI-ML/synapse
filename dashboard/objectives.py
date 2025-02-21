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
            self.__state.objectives[key] = model.evaluate(self.__state.parameters)

    def get(self):
        return self.__state.objectives

    def update(self):
        for key in self.__state.objectives.keys():
            self.__state.objectives[key] = self.__model.evaluate(self.__state.parameters)
        # push again at flush time
        self.__state.dirty("objectives")

    def card(self):
        with v2.VCard(style="width: 500px"):
            with v2.VCardTitle("Objectives"):
                with v2.VCardText():
                    for key in self.__state.objectives.keys():
                        v2.VTextField(
                            v_model_number=(f"objectives['{key}']",),
                            label=key,
                            readonly=True,
                            type="number",
                        )
