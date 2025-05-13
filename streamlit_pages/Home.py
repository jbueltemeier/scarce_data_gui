import io
import os
import pickle
import sys
from typing import cast, List

import streamlit as st

from scarce_data_gui.db import create_mysql_engine

from scarce_data_gui.misc import load_settings
from scarce_data_gui.page_parts import (
    add_funding_notice,
    add_home_image_buttons,
    load_css,
    load_dataset_initialisation,
    remove_sidebar,
)
from scarce_data_gui.streamlit_modules import (
    get_authenticator,
    update_user_auth_config_file,
)

from scarce_data_gui.utils import get_all_current_dataset_filenames, save_dataset


def set_graphviz_path() -> None:
    if sys.platform == "win32":  # Windows
        possible_paths = [
            r"C:\Program Files\Graphviz\bin",
            r"C:\Program Files (x86)\Graphviz\bin",
        ]
    elif sys.platform == "linux":  # Linux
        possible_paths = ["/usr/bin/", "/usr/local/bin/", "/opt/graphviz/bin/"]
    else:
        possible_paths = []

    for path in possible_paths:
        if os.path.exists(path):
            os.environ["PATH"] += os.pathsep + path
            return


set_graphviz_path()

st.set_page_config(page_title="Labeling GUI", page_icon="👋", layout="wide")
remove_sidebar()
load_css()
# -------------------------Initalisation global settings--------------------------------
# load global GUI settings
if "global_settings" not in st.session_state:
    st.session_state["global_settings"] = load_settings()

if "data_handler" in st.session_state:
    del st.session_state["data_handler"]

if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

if "dataset_filename" not in st.session_state:
    st.session_state["dataset_filename"] = ""

# -------------------------Try Starting database----------------------------------------
if "db" not in st.session_state:
    sqlalchemy_database_url = st.session_state.global_settings[
        "sqlalchemy_database_url"
    ]
    if sqlalchemy_database_url is not None:
        st.session_state.db = create_mysql_engine(sqlalchemy_database_url)
    else:
        st.session_state.db = None

# -------------------------Initalisation Home Text--------------------------------------
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.write(st.session_state.global_settings["home_heading"])
    st.write(st.session_state.global_settings["home_gui_description"])
    filenames = get_all_current_dataset_filenames(st.session_state["db"])
    if "dataset_filename" in st.session_state:
        if st.session_state["dataset_filename"]:
            default_index = cast(List[str], filenames).index(
                st.session_state["dataset_filename"]
            )
        else:
            default_index = 0
    else:
        default_index = 1

    col12, col22 = st.columns([0.5, 0.5])
    with col12:
        st.write("## Auswahl Datensatz")
        if filenames is not None:
            st.session_state["dataset_filename"] = st.selectbox(
                "Auswählen des Datensatzes:", filenames, index=default_index
            )
            if st.button("Datensatz laden", type="primary"):
                load_dataset_initialisation()
                st.write("Datensatz erfolgreich geladen!")

        else:
            st.warning("Erstellen Sie zunächst einen Datensatz.")

    with col22:
        st.write("## Bestehenden Datensatz hochladen")
        uploaded_file = st.file_uploader(
            "Lade eine .pkl-Datei hoch. Stellen Sie sicher, dass es sich um eine Datei "
            "aus einem bestehenden Datensatz dieser GUI handelt, da ansonsten die "
            "Funktionalität nicht gewährleistet werden kann.",
            type="pkl",
        )
        if uploaded_file is not None:
            try:
                file_buffer = io.BytesIO(uploaded_file.read())
                dataset = pickle.load(file_buffer)
                load_dataset_initialisation(dataset)
                st.session_state["dataset_filename"] = uploaded_file.name
                st.write("Der Inhalt wurde geladen.")
                save_dataset(
                    dataset=st.session_state["dataset"],
                    filename=st.session_state["dataset_filename"],
                    db=st.session_state["db"],
                )
                st.write(
                    "Erfolgreich gespeichert. Die Seite muss neu geladen werden, um den Datensatz "
                    "auswählen zu können."
                )
            except Exception as e:
                st.error(f"Fehler beim Laden der Datei: {e}")


with col2:
    add_funding_notice()


# -------------------------Authenticator------------------------------------------------
config, st.session_state.authenticator = get_authenticator()

# -------------------------Switch Page Button-------------------------------------------
add_home_image_buttons()
# -------------------------Register new user--------------------------------------------
st.write("## Neuen Benutzer Registrieren")
try:
    if st.session_state.authenticator.register_user(captcha=False):
        st.success("User registered successfully")
except Exception as e:
    st.error(e)
update_user_auth_config_file(config)
