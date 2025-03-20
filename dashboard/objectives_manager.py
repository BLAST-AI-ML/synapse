from trame.widgets import vuetify2 as v2

from state_manager import state

class ObjectivesManager:

    def __init__(self, model, output_variables):
        # FIXME generalize for multiple objectives
        assert len(output_variables) == 1, "number of objectives > 1 not supported"
        # save PyTorch model
        self.__model = model
        # define state variables
        state.objectives = dict()
        for _, objective_dict in output_variables.items():
            key = objective_dict["name"]
            if model.avail():
                state.objectives[key] = model.evaluate(state.parameters)
            else:
                state.objectives[key] = None

    def update(self):
        for key in state.objectives.keys():
            if self.__model.avail():
                state.objectives[key] = self.__model.evaluate(state.parameters)
        # push again at flush time
        state.dirty("objectives")

    def card(self):
        with v2.VCard():
            with v2.VCardTitle("Objectives"):
                with v2.VCardText():
                    for key in state.objectives.keys():
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
