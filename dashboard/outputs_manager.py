from trame.widgets import vuetify3 as vuetify

from state_manager import state


class OutputManager:
    def __init__(self, output_variables):
        print("Initializing output manager...")
        # define state variables
        state.output_variables = [v["name"] for v in output_variables.values()]
        state.displayed_output = state.output_variables[0]

    def panel(self):
        print("Setting output card...")
        with vuetify.VExpansionPanels(v_model=("expand_panel_control_output", 0)):
            with vuetify.VExpansionPanel(
                title="Control: Displayed Output",
                style="font-size: 20px; font-weight: 500;",
            ):
                with vuetify.VExpansionPanelText():
                    with vuetify.VRow():
                        vuetify.VSelect(
                            v_model=("displayed_output",),
                            items=(state.output_variables,),
                            dense=True,
                        )
