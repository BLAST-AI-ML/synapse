from state_manager import state


class ObjectivesManager:
    def __init__(self, model, output_variables):
        print("Initializing objectives manager...")
        # FIXME generalize for multiple objectives
        assert len(output_variables) == 1, "number of objectives > 1 not supported"
        # save model
        self.__model = model
        # define state variables
        state.objectives = dict()
        state.objectives_display_name = dict()
        for objective_dict in output_variables.values():
            key = objective_dict["name"]
            state.objectives_display_name[key] = objective_dict.get("display_name", key)

            if model.avail():
                state.objectives[key], lower, upper = model.evaluate(state.parameters)
            else:
                state.objectives[key] = None
