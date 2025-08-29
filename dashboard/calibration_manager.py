from state_manager import state
import copy

class SimulationCalibrationManager:
    def __init__(self, simulation_calibration):
        state.simulation_calibration = copy.deepcopy(simulation_calibration)

    def convert_sim_to_exp(self, df_sim):
        """
        Apply calibration to the simulation points, so as to reconstruct
        the same input/output variables as the experimental points
        """
        for value in state.simulation_calibration.values():
            sim_name = value["name"]
            exp_name = value["depends_on"]
            df_sim[exp_name] = df_sim[sim_name] / value["alpha"] + value["beta"]
