import inspect
import numpy as np
from scipy.optimize import minimize
import sys
from lume_model.models.torch_model import TorchModel
from lume_model.models.gp_model import GPModel
from state_manager import state


class ModelManager:


    def __init__(self, model_data):
        print(f"Initializing model manager...")
        if model_data is None:
            self.__model = None
        else:
            self.__is_neural_network = False
            self.__is_gaussian_process = False
            try:
                if state.model_type == "Neural Network":
                    self.__is_neural_network = True
                    self.__model = TorchModel(model_data)
                elif state.model_type == "Gaussian Process":
                    self.__is_gaussian_process = True
                    self.__model = GPModel.from_yaml(model_data)
                else:
                    raise ValueError(f"Unsupported model type: {state.model_type}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                sys.exit(1)


    def avail(self):
        print("Checking model availability...")
        model_avail = True if self.__model is not None else False
        return model_avail


    @property
    def is_neural_network(self):
        return self.__is_neural_network


    @property
    def is_gaussian_process(self):
        return self.__is_gaussian_process


    def evaluate(self, parameters_model):
        print("Evaluating model...")
        if self.__model is not None:
            # evaluate model
            output_dict = self.__model.evaluate(parameters_model)
            if self.__is_neural_network:
                # expected only one value
                if len(output_dict.values()) != 1:
                    raise ValueError(f"Expected 1 output value, but found {len(output_dict.values())}")
                mean = list(output_dict.values())[0]
                mean_error = 0.0  # trick to collapse error range when lower/upper bounds are not predicted
                lower = mean - mean_error
                upper = mean + mean_error
            elif self.__is_gaussian_process:
                # TODO use "exp" only once experimental data is available for all experiments
                task_tag = "exp" if state.experiment == "ip2" else "sim"
                output_key = [key for key in output_dict.keys() if task_tag in key][0]
                mean = output_dict[output_key].mean
                mean_error = 2.*output_dict[output_key].variance.sqrt()
                lower = mean - mean_error
                upper = mean + mean_error
                # detach gradients from tensor
                mean = mean.detach()
                lower = lower.detach()
                upper = upper.detach()
            else:
                raise ValueError(f"Unsupported model type: {state.model_type}")
            # convert to Python float if tensor has only one element
            # because Trame state variables must be serializable
            if mean.numel() == 1:
                mean = float(mean)
            return (mean, lower, upper)


    def model_wrapper(self, parameters_array):
        print("Wrapping model...")
        # convert array of parameters to dictionary
        parameters_dict = dict(zip(state.parameters.keys(), parameters_array))
        # change sign to the result in order to maximize when optimizing
        mean, lower, upper = self.evaluate(parameters_dict)
        res = -mean
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
