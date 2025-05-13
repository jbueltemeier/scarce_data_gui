import streamlit as st


__all__ = ["remove_sidebar"]


def remove_sidebar() -> None:
    hide_sidebar_style = """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
    """
    st.markdown(hide_sidebar_style, unsafe_allow_html=True)
