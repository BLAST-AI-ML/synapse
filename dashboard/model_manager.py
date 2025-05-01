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
        # inspect current function and module names
        cfunct = inspect.currentframe().f_code.co_name
        cmodul = os.path.basename(inspect.currentframe().f_code.co_filename)
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
                print(f"{cmodul}:{self.__class__.__name__}.{cfunct}: {e}")
                sys.exit(1)

    def avail(self):
        model_avail = True if self.__model is not None else False
        return model_avail

    def posterior(self, tensor_combined):
        if self.__model is not None:
            predictions = self.__model.posterior(tensor_combined)
            with torch.no_grad():
                mean = predictions.mean
                l,u = predictions.mvn.confidence_region()
                variance = predictions.variance

            res = mean[:,1].detach().cpu().numpy().tolist()
            return res, l[:,1].detach().cpu().numpy().tolist(), u[:,1].detach().cpu().numpy().tolist()

    def evaluate(self, parameters_model):
        if self.__model is not None:

            # evaluate model
            output_dict = self.__model.evaluate(parameters_model)
            if state.model_type == "NN":
                #expected only one value
                if len(output_dict.values()) != 1:
                    raise ValueError(f"Expected 1 output value, but found {len(output_dict.values())}")
                res = list(output_dict.values())[0]
                #convert to Python float if tensor has only one element (more elements for line plots)
                if res.numel() == 1:
                    res = float(res)
                return res
            elif state.model_type == "GP":
                mean = output_dict['n_protons_exp_task'].mean
                l, u = (
                    mean - 2. * output_dict['n_protons_exp_task'].variance.sqrt(),
                    mean + 2. * output_dict['n_protons_exp_task'].variance.sqrt(),
                )
                return mean.detach().numpy().tolist(), l.detach().numpy().tolist(), u.detach().numpy().tolist()
            else:
                raise ValueError(f"Unsupported model_type: {state.model_type}")

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

    def get_output_transformers(self):
        if self.__model is not None:
            return self.__model.output_transformers