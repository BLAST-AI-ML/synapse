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
