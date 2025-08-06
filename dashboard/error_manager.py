from trame.widgets import vuetify3 as vuetify, html
from state_manager import state


def add_error(title, msg):
    state.errors.append(
        {
            "id": state.error_counter,
            "title": title,
            "msg": msg,
        }
    )
    state.error_counter += 1
    state.dirty("errors")


def remove_error(i):
    state.errors.pop(int(i))
    state.dirty("errors")


def remove_all_errors():
    state.errors = []
    state.dirty("errors")


def error_panel():
    with vuetify.VExpansionPanels(
        v_if=("errors.length != 0",),
        dense=True,
        popout=True,
        style="width: 350px; z-index: 20000; position: fixed; bottom: 16px; left: 16px;",
    ):
        with vuetify.VExpansionPanel(
            dense=True,
            title=("errors.length + ' Errors Generated'",),
            expand_icon="mdi-alert",
            disable_icon_rotate=True,
            color="error",
        ):
            with vuetify.VExpansionPanelText(
                dense=True,
            ):
                with vuetify.VAlert(
                    v_for="(alert, i) in errors",
                    key="alert.id",
                    dense=True,
                    closeable=True,
                    close_icon=("mdi-close",),
                    input=(remove_error, "[i]"),
                ):
                    html.H4("{{alert.title}}")
                    html.P("{{alert.msg}}")
                vuetify.VBtn(
                    "Close all",
                    dense=True,
                    click=remove_all_errors,
                    color="error",
                    variant="text",
                    block=True,
                )
