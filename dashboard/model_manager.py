import numpy as np
from scipy.optimize import minimize
import torch

from lume_model.models.torch_model import TorchModel

from state_manager import state

class ModelManager:

    def __init__(self, model_data):
        if model_data is None:
            self.__model = None
        else:
            try:
                self.__model = TorchModel(model_data)
            except Exception as e:
                print(f"ModelManager.__init__: {e}")
                sys.exit(1)

    def avail(self):
        model_avail = True if self.__model is not None else False
        return model_avail

    def evaluate(self, parameters_model):
        if self.__model is not None:
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
        else:
            return None

    def model_wrapper(self, parameters_array):
        # convert array of parameters to dictionary
        parameters_dict = dict(zip(state.parameters.keys(), parameters_array))
        # change sign to the result in order to maximize when optimizing
        res = -self.evaluate(parameters_dict)
        return res

    def optimize(self):
        if self.__model is not None:
            # get array of current parameters from state
            parameters_values = np.array(list(state.parameters.values()))
            # define parameters bounds for optimization
            parameters_bounds = []
            for key in state.parameters.keys():
                parameters_bounds.append((state.parameters_min[key], state.parameters_max[key]))
            # optimize model (maximize output value)
            res = minimize(
                fun=self.model_wrapper,
                x0=parameters_values,
                bounds=parameters_bounds,
            )
            # update parameters in state with optimal values
            state.parameters = dict(zip(state.parameters.keys(), res.x))
            # push again at flush time
            state.dirty("parameters")
        else:
            return

    def get_output_transformers(self):
        if self.__model is not None:
            return self.__model.output_transformers
        else:
            return None
