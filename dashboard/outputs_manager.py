import copy
from trame.widgets import vuetify2 as vuetify

from state_manager import state


class OutputManager:
    def __init__(self, output_variables):
        print("Initializing output manager...")
        # define state variables
        self.output_variables = [ v['name'] for v in output_variables.values() ]
        state.displayed_output = self.output_variables[0]

    def panel(self):
        print("Setting output card...")
        print(self.output_variables)
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_output", 0)):
            with vuetify.VExpansionPanel():
                vuetify.VExpansionPanelHeader(
                    "Control: Displayed Output", style="font-size: 20px; font-weight: 500;"
                )
                with vuetify.VExpansionPanelContent():
                    # create a row for the switches and buttons
                    with vuetify.VRow():
                        with vuetify.VCol():
                            vuetify.VSelect(
                                v_model=("displayed_output",),
                                items=("Output", self.output_variables),
                                dense=True,
                                style="margin-left: 16px; margin-top: 24px; max-width: 210px;",
                            )
