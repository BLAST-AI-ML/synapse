import numpy as np
from scipy.optimize import minimize
from trame.widgets import vuetify3 as vuetify

from error_manager import add_error
from state_manager import state


class OptimizationManager:
    def __init__(self, model):
        print("Initializing optimization manager...")
        self.__model = model
        state.optimization_target = state.displayed_output

    def model_wrapper(self, parameters_array):
        print("Wrapping model...")
        # convert array of parameters to dictionary
        parameters_dict = dict(zip(state.parameters.keys(), parameters_array))
        # change sign to the result in order to maximize when optimizing
        mean, lower, upper = self.__model.evaluate(
            parameters_dict, state.optimization_target
        )
        res = -mean if state.optimization_type == "Maximize" else mean
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
                method="Powell",
            )
            print(f"Optimization result:\n{res}")
            # update parameters in state with optimal values
            state.parameters = dict(zip(state.parameters.keys(), res.x))
            # push again at flush time
            state.dirty("parameters")
            # Force flush now (TODO fix state change listeners, remove workaround)
            state.flush()
            # update optimization status
            if res.success:
                state.optimization_status = "Completed"
            else:
                state.optimization_status = "Failed"
                title = "Unable to optimize parameters"
                msg = f"Error occurred when optimizing parameters: {res.message}"
                add_error(title, msg)

    def optimize_trigger(self):
        try:
            self.optimize()
        except Exception as e:
            title = "Unable to optimize parameters"
            msg = f"Error occurred when optimizing parameters: {e}"
            add_error(title, msg)
            print(msg)

    def panel(self):
        print("Setting optimization card...")
        # list of available optimization operations
        optimization_type_list = [
            "Maximize",
            "Minimize",
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
                                v_model=("optimization_target",),
                                label="Optimization target",
                                items=(state.output_variables,),
                                dense=True,
                            )
                        with vuetify.VCol():
                            vuetify.VSelect(
                                v_model=("optimization_type",),
                                label="Optimization type",
                                items=(optimization_type_list,),
                                dense=True,
                            )
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VBtn(
                                "Optimize",
                                click=self.optimize_trigger,
                                style="text-transform: none",
                            )
