from trame.widgets import vuetify3 as vuetify

from state_manager import state


class OutputsManager:
    def __init__(self, outputs):
        print("Initializing output manager...")
        # define state variables
        state.outputs = [v["name"] for v in outputs.values()]
        state.displayed_output = state.outputs[0]

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
                            items=(state.outputs,),
                            dense=True,
                        )
