import pandas as pd

import streamlit as st

from scarce_data_gui.page_parts import (
    add_dataset_save_button,
    add_funding_notice,
    add_optimise_button,
    add_redirection_buttons,
    load_css,
    remove_sidebar,
)


st.set_page_config(layout="wide")
remove_sidebar()

load_css()

col1, col2 = st.columns([0.6, 0.4])
with col1:
    add_redirection_buttons(exclude="Dataset")

with col2:
    add_funding_notice()

col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
with col1:
    st.subheader("Datensatz optimieren")
    labels = st.session_state["dataset"].user_label_container.combine_labels()
    if not (labels == "unlabeled").all().all():
        warning = st.warning(
            "Achtung! Die Label werden durch diese Aktion nicht mehr gültig?",
            icon="⚠️",
        )
        confirmation = st.checkbox("Ja, ich möchte fortfahren")
    else:
        confirmation = True
    add_optimise_button(confirmation=confirmation)

with col2:
    st.subheader("Speichern")
    value = (
        st.session_state["dataset_filename"]
        if "dataset_filename" in st.session_state
        else ""
    )
    filename = st.text_input("Name des Datensatzes", value=value)
    add_dataset_save_button(filename=filename)
with col3:
    st.subheader("Ploteinstellungen Achsen")
    current_df = st.session_state["dataset"].get_dataframe(remove_extension=False)
    numerical_columns = [col for col in current_df.columns if "_numerical" in col]

    x_axis_feature: str = st.selectbox(
        "Wählen Sie das Merkmal für die x-Achse':",
        options=numerical_columns,
        index=0,
    )
    y_axis_feature: str = st.selectbox(
        "Wählen Sie das Merkmal für die y-Achse':",
        options=numerical_columns,
        index=1,
    )
with col4:
    st.subheader("Ploteinstellungen Labelvisualisierung")
    options = list(labels.columns)
    options.insert(0, "None")
    label_choice = st.selectbox(
        "Wählen Sie eine Ausgangsgröße:", options=options, index=0
    )


col1, _, col3 = st.columns([0.45, 0.1, 0.45])
with col1:
    st.header("Initialer Datensatz")
    st.plotly_chart(
        st.session_state["dataset"].plot_input_space(
            initial=True, x_axis_feature=x_axis_feature, y_axis_feature=y_axis_feature
        ),
        key="initial_dataset",
    )
    df = st.session_state["dataset"].get_dataframe(initial=True)
    st.dataframe(df, use_container_width=True)

with col3:
    st.header("Optimierte Datensatz")
    if label_choice == "None":
        labels = None
    else:
        labels = st.session_state["dataset"].get_labels()
        labels = labels[[label_choice]]

    st.plotly_chart(
        st.session_state["dataset"].plot_input_space(
            active_learning_mode=True,
            x_axis_feature=x_axis_feature,
            y_axis_feature=y_axis_feature,
            labels=labels,
        ),
        key="optimised_dataset",
    )
    df = st.session_state["dataset"].get_dataframe()
    labels = st.session_state["dataset"].user_label_container.combine_labels()
    if st.session_state["dataset"].active_learning_label_container is not None:
        active_learning_labels = st.session_state[
            "dataset"
        ].active_learning_label_container.combine_labels()
        labels = pd.concat([labels, active_learning_labels], ignore_index=True)
    df = pd.concat([df, labels], axis=1)
    st.dataframe(df.astype(str), use_container_width=True)

# Upload excel data (be careful with this)
# uploaded_file = st.file_uploader(
#     "Laden Sie die Daten aus einer Excel-Datei in den Datensatz",
#     type=["xlsx", "xls"],
# )
# warning_2 = st.warning(
#     "Achtung! Die Label werden durch diese Aktion nicht mehr gültig?",
#     icon="⚠️",
# )
# confirmation = st.checkbox("Ja, die Daten überschreiben!")
# if uploaded_file is not None and confirmation:
#     try:
#         new_df = pd.read_excel(uploaded_file)
#         st.session_state["dataset"].overwrite_df(new_df)
#     except Exception as e:
#         st.error(f"Fehler beim Lesen der Datei: {e}")
