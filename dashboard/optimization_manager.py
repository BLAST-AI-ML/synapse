import numpy as np
from scipy.optimize import minimize
from trame.widgets import vuetify3 as vuetify

from state_manager import state


class OptimizationManager:
    def __init__(self, model):
        print("Initializing optimization manager...")
        self.__model = model

    def model_wrapper(self, parameters_array):
        print("Wrapping model...")
        # convert array of parameters to dictionary
        parameters_dict = dict(zip(state.parameters.keys(), parameters_array))
        # change sign to the result in order to maximize when optimizing
        mean, lower, upper = self.__model.evaluate(parameters_dict)
        res = -mean
        return res

    def optimize(self):
        print("Optimizing parameters...")
        # info print statement skipped to avoid redundancy
        if self.__model is not None:
            # get array of current parameters from state
            parameters_values = np.array(list(state.parameters.values()))
            # define parameters bounds for optimization
            parameters_bounds = []
            for key in state.parameters.keys():
                parameters_bounds.append(
                    (state.parameters_min[key], state.parameters_max[key])
                )
            # optimize model (maximize output value)
            res = minimize(
                fun=self.model_wrapper,
                x0=parameters_values,
                bounds=parameters_bounds,
                method="Nelder-Mead",
            )
            # update parameters in state with optimal values
            state.parameters = dict(zip(state.parameters.keys(), res.x))
            # update optimization status
            state.optimization_status = "Success" if res.success else "Failed"
            # push again at flush time
            state.dirty("parameters")
            state.flush()

    def panel(self):
        print("Setting optimization card...")
        # list of available optimization operations
        optimization_type_list = [
            "Maximize",
        ]
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_optimization", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Optimization",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText():
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VSelect(
                                v_model=("optimization_type",),
                                label="Optimization type",
                                items=("optimization", optimization_type_list),
                                dense=True,
                            )
                        with vuetify.VCol():
                            vuetify.VTextField(
                                v_model_number=("optimization_status",),
                                label="Optimization status",
                                readonly=True,
                            )
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VBtn(
                                "Optimize",
                                click=self.optimize,
                                style="text-transform: none",
                            )
