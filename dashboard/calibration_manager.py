from trame.widgets import client, vuetify3 as vuetify
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

    def panel(self):
        print("Setting calibration card...")
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_calibration", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Calibrate simulation points",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText():
                    with client.DeepReactive("simulation_calibration"):
                        for count, key in enumerate(state.simulation_calibration.keys()):
                            # create a row for the parameter label
                            with vuetify.VRow():
                                vuetify.VListSubheader(
                                    state.simulation_calibration[key]["name"],
                                    style=(
                                        "margin-top: 16px;"
                                        if count == 0
                                        else "margin-top: 0px;"
                                    ),
                                )
                            with vuetify.VRow(no_gutters=True):
                                with vuetify.VCol():
                                    vuetify.VTextField(
                                        v_model_number=(f"simulation_calibration['{key}']['alpha']",),
                                        change=f"flushState('simulation_calibration')",
                                        density="compact",
                                        hide_details=True,
                                        style="width: 100px;",
                                        type="number",
                                        label="alpha",
                                    )
                                with vuetify.VCol():
                                    vuetify.VTextField(
                                        v_model_number=(f"simulation_calibration['{key}']['beta']",),
                                        change=f"flushState('simulation_calibration')",
                                        density="compact",
                                        hide_details=True,
                                        style="width: 100px;",
                                        type="number",
                                        label="beta",
                                    )
