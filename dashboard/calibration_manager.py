class SimulationCalibrationManager:
    def __init__(self, simulation_calibration):
        self.simulation_calibration = simulation_calibration

    def convert_sim_to_exp(self, df_sim):
        """
        Apply calibration to the simulation points, so as to reconstruct
        the same input/output variables as the experimental points
        """
        for value in self.simulation_calibration.values():
            sim_name = value["name"]
            exp_name = value["depends_on"]
            df_sim[exp_name] = df_sim[sim_name] / value["alpha"] + value["beta"]

    def convert_exp_to_sim(self, exp_dict):
        """
        Apply calibration to the experimental points, to be passed as
        parameters for simulations on NERSC
        """
        sim_dict = {}
        for _, value in self.simulation_calibration.items():
            sim_name = value["name"]
            exp_name = value["depends_on"]
            # strip characters after '[' parenthesis to remove units, strip
            # leading/trailing white spaces, replace white spaces and '-' with '_',
            # and convert to lower case
            sim_name = (
                sim_name.split("[")[0]
                .strip()
                .replace(" ", "_")
                .replace("-", "_")
                .lower()
            )
            # fill the dictionary
            if exp_name in exp_dict:
                sim_val = (exp_dict[exp_name] - value["beta"]) * value["alpha"]
                sim_dict[sim_name] = sim_val
        return sim_dict
