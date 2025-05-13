import streamlit as st
from scarce_data_gui.page_parts import remove_sidebar

from streamlit_extras.switch_page_button import switch_page


st.set_page_config(page_title="Labeling GUI", page_icon="👋")
remove_sidebar()

# -------------------------Login------------------------------------------
st.session_state.authenticator.login("main")


# -------------------------Check Login--------------------------------------------------
if st.session_state["authentication_status"]:
    if "dataset_filename" in st.session_state:
        switch_page("Labeling")
    else:
        # TODO: find better solution important when no files uploaded
        st.session_state.authenticator.logout(location="main")
        st.warning(
            "Datenfile existiert nicht in der Datenbank. Bitte die Schreibweise "
            "korrigieren oder das File hochladen."
        )

elif st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")
elif st.session_state["authentication_status"] is None:
    st.warning("Please enter your username and password")
