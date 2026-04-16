from trame.widgets import vuetify3 as vuetify


def load_iriapi_card():
    print("Setting AmSC IRI API card...")
    # row with component to upload input file with top padding
    with vuetify.VRow(style="padding-top: 20px;"):
        with vuetify.VCol():
            vuetify.VFileInput(
                v_model=("sfapi_key_dict",),
                label="Token File",
                accept=".pem",
                prepend_icon="",
                prepend_inner_icon="mdi-paperclip",
                __properties=["accept"],
            )
    # row with text field to display key expiration date
    with vuetify.VRow():
        with vuetify.VCol():
            vuetify.VTextField(
                v_model=("sfapi_key_expiration",),
                label="Token Expiration (if expired or unavailable, please upload a valid token)",
                readonly=True,
            )
    # row with text field to display Perlmutter status
    with vuetify.VRow():
        with vuetify.VCol():
            vuetify.VTextField(
                v_model=("perlmutter_description",),
                label="Perlmutter Status",
                readonly=True,
            )
