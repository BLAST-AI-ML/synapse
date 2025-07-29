import copy
from trame.widgets import vuetify2 as vuetify

from state_manager import state

class SimulationCalibrationManager:

    def __init__(self, simulation_calibration):
        self.simulation_calibration = simulation_calibration

    def convert_sim_to_exp( self, df_sim ):
        """
        Apply calibration to the simulation points, so as to reconstruct 
        the same input/output variables as the experimental points
        """
        for value in self.simulation_calibration.values():
            sim_name = value["name"]
            exp_name = value["depends_on"]
            df_sim[exp_name] = df_sim[sim_name] / value["alpha"] + value["beta"]

    def convert_exp_to_sim(self, exp_dict, sim_dict):
        """
        Apply calibration to the experimental points, to be passed as 
        parameters for simulations on NERSC
        """
        for sim_name, values in self.simulation_calibration.items():
            exp_name = values["depends_on"]
            if exp_name in exp_dict:
                sim_val = (exp_dict[exp_name] - values["beta"]) * values["alpha"]
                sim_dict["var_name"].append(exp_name)
                sim_dict["sim_val"].append(sim_val)

        return sim_dict
