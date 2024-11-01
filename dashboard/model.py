import numpy as np
from scipy.optimize import minimize
import torch

from lume_model.models import TorchModel

class Model:
    def __init__(self, server, model_file):
        # Trame state and controller
        self.__state = server.state
        self.__ctrl = server.controller
        # PyTorch model
        self.__model = TorchModel(model_file)
        # decorate method 'optimize'
        self.optimize = self.__ctrl.add("optimize")(self.optimize)

    def format(self, parameters):
        parameters_model = parameters.copy()
        # workaround to match keys:
        # - model labels do not carry units (e.g., "TOD" instead of "TOD (fs^3)")
        for key_old in parameters_model.keys():
            key_new, _ = key_old.split(maxsplit=1)
            parameters_model[key_new] = parameters_model.pop(key_old)
        # - model inputs do not include GVD
        gvd_key = [key_tmp for key_tmp in parameters_model.keys() if key_tmp == "GVD"][0]
        parameters_model.pop(gvd_key)
        return parameters_model

    def evaluate(self, parameters_model):
        # evaluate model
        output_dict = self.__model.evaluate(parameters_model)
        # expected only one value
        if len(output_dict.values()) != 1:
            raise ValueError(f"Expected 1 output value, but found {len(output_dict.values())}")
        res = list(output_dict.values())[0]
        # convert to Python float if tensor has only one element (more elements for line plots)
        if res.numel() == 1:
            res = float(res)
        return res

    def _model_wrapper(self, parameters_array):
        # convert array of parameters to dictionary
        parameters_dict = dict(zip(self.__state.parameters.keys(), parameters_array))
        # format dictionary of parameters for model evaluation
        parameters_model = self.format(parameters_dict)
        # change sign to the result in order to maximize when optimizing
        res = -self.evaluate(parameters_model)
        return res

    def optimize(self):
        # get array of current parameters from state
        parameters_values = np.array(list(self.__state.parameters.values()))
        # define parameters bounds for optimization
        parameters_bounds = []
        for key in self.__state.parameters.keys():
            parameters_bounds.append((self.__state.parameters_min[key], self.__state.parameters_max[key]))
        # optimize model (maximize output value)
        res = minimize(
            fun=self._model_wrapper,
            x0=parameters_values,
            bounds=parameters_bounds,
        )
        # update parameters in state with optimal values 
        self.__state.parameters = dict(zip(self.__state.parameters.keys(), res.x))
        # push again at flush time
        self.__state.dirty("parameters")
