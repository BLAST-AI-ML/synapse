import numpy as np
from scipy.optimize import minimize
import torch

from lume_model.models.torch_model import TorchModel

class Model:
    def __init__(self, server, model_data):
        # Trame state and controller
        self.__state = server.state
        self.__ctrl = server.controller
        if model_data is None:
            print(f"Model.__init__: Model not provided, skip initialization")
            self.__model = None
        else:
            try:
                self.__model = TorchModel(model_data)
            except Exception as e:
                print(f"Model.__init__: {e}")
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
            print(f"Model.evaluate: Model not provided, skip evaluation")
            return None

    def model_wrapper(self, parameters_array):
        # convert array of parameters to dictionary
        parameters_dict = dict(zip(self.__state.parameters.keys(), parameters_array))
        # change sign to the result in order to maximize when optimizing
        res = -self.evaluate(parameters_dict)
        return res

    def optimize(self):
        if self.__model is not None:
            # get array of current parameters from state
            parameters_values = np.array(list(self.__state.parameters.values()))
            # define parameters bounds for optimization
            parameters_bounds = []
            for key in self.__state.parameters.keys():
                parameters_bounds.append((self.__state.parameters_min[key], self.__state.parameters_max[key]))
            # optimize model (maximize output value)
            res = minimize(
                fun=self.model_wrapper,
                x0=parameters_values,
                bounds=parameters_bounds,
            )
            # update parameters in state with optimal values
            self.__state.parameters = dict(zip(self.__state.parameters.keys(), res.x))
            # push again at flush time
            self.__state.dirty("parameters")
        else:
            print(f"Model.optimize: Model not provided, skip optimization")
            return

    def get_output_transformers(self):
        if self.__model is not None:
            return self.__model.output_transformers
        else:
            print(f"Model.get_output_transformers: Model not provided, skip transformers")
            return None
