from state_manager import state


class OutputManager:
    def __init__(self, output_variables):
        print("Initializing output manager...")
        # define state variables
        state.output_variables = [v["name"] for v in output_variables.values()]
        state.displayed_output = state.output_variables[0]
