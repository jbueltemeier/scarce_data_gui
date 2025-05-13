import streamlit as st
from scarce_data_gui.page_parts import (
    add_funding_notice,
    add_redirection_buttons,
    display_data_acquisition_slider,
    display_results,
    load_css,
    remove_sidebar,
)

# -------------------------Initalisation Page-------------------------------------------
st.set_page_config(layout="wide")
remove_sidebar()

load_css()
col1, col2 = st.columns([0.6, 0.4])
with col1:
    add_redirection_buttons(exclude="Labeling")

with col2:
    add_funding_notice()

# -------------------------Show Columns-------------------------------------------------
col1, col2 = st.columns([3, 1], gap="large")
display_results(col1)
display_data_acquisition_slider(col2)
