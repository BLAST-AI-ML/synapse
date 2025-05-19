import inspect
import numpy as np
import os
from scipy.optimize import minimize
import sys
import torch
from lume_model.models.torch_model import TorchModel
from lume_model.variables import ScalarVariable, DistributionVariable
from lume_model.models.gp_model import GPModel
from state_manager import state
import os

class ModelManager:

    def __init__(self, model_data):
        print(f"Initializing model manager...")
        if model_data is None:
            self.__model = None
        else:
            try:
                if state.model_type == "NN":
                    self.__model = TorchModel(model_data)
                elif state.model_type == "GP":
                    self.__model = GPModel.from_yaml(model_data)
                else:
                    raise ValueError(f"Unsupported model_type: {state.model_type}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                sys.exit(1)

    def avail(self):
        print("Checking model availability...")
        model_avail = True if self.__model is not None else False
        return model_avail

    def evaluate(self, parameters_model):
        print("Evaluating model...")
        if self.__model is not None:

            # evaluate model
            output_dict = self.__model.evaluate(parameters_model)
            if state.model_type == "NN":
                #expected only one value
                if len(output_dict.values()) != 1:
                    raise ValueError(f"Expected 1 output value, but found {len(output_dict.values())}")
                res = list(output_dict.values())[0]
                #convert to Python float if tensor has only one element (more elements for line plots)
                if isinstance(res, torch.Tensor) and res.numel() == 1:
                    res = float(res)
                return res
            elif state.model_type == "GP":
                if state.experiment == 'ip2':
                    output_key = next((key for key in output_dict if 'exp' in key), None)
                elif state.experiment in ['acave', 'qed_ip2']:
                    output_key = next((key for key in output_dict if 'sim' in key), None)
                mean = output_dict[output_key].mean
                l, u = (
                    mean - 2. * output_dict[output_key].variance.sqrt(),
                    mean + 2. * output_dict[output_key].variance.sqrt(),
                )
                return mean.detach().numpy().tolist(), l.detach().numpy().tolist(), u.detach().numpy().tolist()
            else:
                raise ValueError(f"Unsupported model_type: {state.model_type}")

    def model_wrapper(self, parameters_array):
        print("Wrapping model...")
        # convert array of parameters to dictionary
        parameters_dict = dict(zip(state.parameters.keys(), parameters_array))
        # change sign to the result in order to maximize when optimizing
        res = -self.evaluate(parameters_dict)
        return res

    def optimize(self):
        # info print statement skipped to avoid redundancy
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

    def get_output_transformers(self):
        print("Getting output transformers...")
        if self.__model is not None:
            return self.__model.output_transformers