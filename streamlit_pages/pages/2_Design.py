import streamlit as st

from scarce_data_gui.page_parts import (
    add_design_specific_button,
    add_funding_notice,
    add_redirection_buttons,
    data_scrollable_container,
    load_css,
    remove_sidebar,
)
from scarce_data_gui.utils import INSTANCE_META, INSTANCE_PARAMS, INSTANCE_SETTINGS

if "active_rework" not in st.session_state:
    st.session_state["active_rework"] = False

st.set_page_config(layout="wide")
remove_sidebar()

load_css()
col1, col2 = st.columns([0.6, 0.4])
with col1:
    add_redirection_buttons(exclude="Design")

    add_design_specific_button()
with col2:
    add_funding_notice()


col1, col2 = st.columns([0.7, 0.3])
with col1:
    data_scrollable_container(header=INSTANCE_META)

with col2:
    data_scrollable_container(header=INSTANCE_SETTINGS)

data_scrollable_container(header=INSTANCE_PARAMS)
