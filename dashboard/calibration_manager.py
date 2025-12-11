from trame.widgets import client, vuetify3 as vuetify, html
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

    def convert_exp_to_sim(self, exp_dict):
        """
        Apply calibration to the experimental points, to be passed as
        parameters for simulations on NERSC
        """
        sim_dict = {}
        for _, value in state.simulation_calibration.items():
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

    def panel(self):
        print("Setting calibration card...")
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_calibration", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Calibrate simulation points",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText(
                    style="font-weight: lighter; font-size: 16px;"
                ):
                    with client.DeepReactive("simulation_calibration"):
                        for key in state.simulation_calibration.keys():
                            # create a row for the parameter label
                            with vuetify.VRow():
                                html.Small(
                                    f"<b> {state.simulation_calibration[key]['name']}</b> = α × (<b>{state.simulation_calibration[key]['depends_on']}</b> - β)",
                                )
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    html.Small("α = ", style="width: 100px;")
                                with vuetify.VCol():
                                    vuetify.VTextField(
                                        v_model_number=(
                                            f"simulation_calibration['{key}']['alpha']",
                                        ),
                                        change="flushState('simulation_calibration')",
                                        density="compact",
                                        hide_details=True,
                                        hide_spin_buttons=True,
                                        style="width: 100px;",
                                        type="number",
                                    )
                                with vuetify.VCol():
                                    html.Small("±")
                                with vuetify.VCol():
                                    vuetify.VTextField(
                                        v_model_number=(
                                            f"simulation_calibration['{key}']['alpha_uncertainty']",
                                        ),
                                        density="compact",
                                        hide_details=True,
                                        hide_spin_buttons=True,
                                        style="width: 100px;",
                                        type="number",
                                    )
                            with vuetify.VRow():
                                with vuetify.VCol():
                                    html.Small("β = ")
                                with vuetify.VCol():
                                    vuetify.VTextField(
                                        v_model_number=(
                                            f"simulation_calibration['{key}']['beta']",
                                        ),
                                        change="flushState('simulation_calibration')",
                                        density="compact",
                                        hide_details=True,
                                        hide_spin_buttons=True,
                                        style="width: 100px;",
                                        type="number",
                                    )
                                with vuetify.VCol():
                                    html.Small("±", style="width: 100px;")
                                with vuetify.VCol():
                                    vuetify.VTextField(
                                        v_model_number=(
                                            f"simulation_calibration['{key}']['beta_uncertainty']",
                                        ),
                                        density="compact",
                                        hide_details=True,
                                        hide_spin_buttons=True,
                                        style="width: 100px;",
                                        type="number",
                                    )
