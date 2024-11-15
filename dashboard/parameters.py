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
