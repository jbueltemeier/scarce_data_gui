from typing import cast, TYPE_CHECKING

import streamlit as st

from experimental_design.dataset import PartDesignDataset
from streamlit_extras.switch_page_button import switch_page

from scarce_data_gui.utils import save_dataset

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

__all__ = [
    "display_data_acquisition_slider",
    "display_active_learning_slider",
]


def speed_up_labeling_example_dataset() -> None:
    from example.oven_model import get_complete_label_dict  # type: ignore

    df = st.session_state["dataset"].get_dataframe()
    sub_part_label_keys = list(
        st.session_state["dataset"].user_label_container.sub_label_parts.keys()
    )
    label_dict = get_complete_label_dict(df, sub_part_label_keys)
    st.session_state["dataset"].user_label_container.update_all_labels(
        st.session_state["name"], label_dict
    )
    st.write("Ofendatensatz erfolgreich initialisiert.")
    st.rerun()


def previous_button_click() -> None:
    st.session_state["dataset"].user_label_container.change_position(
        user_name=st.session_state["name"], value=-1
    )
    save_dataset(
        st.session_state["dataset"],
        st.session_state["dataset_filename"],
        db=st.session_state["db"],
    )


def next_button_click() -> None:
    st.session_state["dataset"].user_label_container.change_position(
        user_name=st.session_state["name"], value=1
    )
    save_dataset(
        st.session_state["dataset"],
        st.session_state["dataset_filename"],
        db=st.session_state["db"],
    )


def display_data_acquisition_slider(col: "DeltaGenerator") -> None:
    with col:
        if st.session_state.get("authentication_status") is None:
            switch_page("Home")
        else:
            col_1, col_2 = st.columns(2)
            with col_1:
                st.write(f'Eingeloggt als  *{st.session_state["name"]}*')
                st.session_state["authenticator"].logout("Logout")
            with col_2:
                if "oven" in st.session_state["dataset_filename"]:
                    if st.button("Ofen Speed Button"):
                        speed_up_labeling_example_dataset()

        st.header("Label")
        cast(
            PartDesignDataset, st.session_state["dataset"]
        ).user_label_container.display_sliders(user_name=st.session_state["name"])

        position = (
            st.session_state["dataset"].user_label_container.position(
                user_name=st.session_state["name"]
            )
            + 1
        )
        max_position = st.session_state["dataset"].user_label_container.num_samples
        progress_text = f"Fortschritt: {position}/{max_position}"
        st.progress(value=position / max_position, text=progress_text)

        col1, col2 = st.columns([1, 1])
        with col1:
            if position != 1:
                st.button(
                    "Zurück",
                    on_click=previous_button_click,
                    use_container_width=True,
                    type="primary",
                )

        with col2:
            if position != max_position:
                st.button(
                    "Nächste",
                    on_click=next_button_click,
                    use_container_width=True,
                    type="primary",
                )


def generate_next_point() -> None:
    active_learning_labels = st.session_state[
        "dataset"
    ].active_learning_label_container.combine_labels()
    if not (active_learning_labels == "unlabeled").any().any():
        with st.spinner("Berechnung läuft..."):
            st.session_state["dataset"].perform_active_learning_step()
        st.session_state["dataset"].active_learning_label_container.change_position(
            user_name="active_learning", value=1
        )
        save_dataset(
            st.session_state["dataset"],
            st.session_state["dataset_filename"],
            db=st.session_state["db"],
        )
    else:
        st.warning("Label zunächst diesen Datenpunkt.")


def display_active_learning_slider(col: "DeltaGenerator") -> None:
    with col:
        st.header("Label")
        cast(  # type: ignore[union-attr]
            PartDesignDataset, st.session_state["dataset"]
        ).active_learning_label_container.display_sliders(user_name="active_learning")
        st.button(
            "Generate",
            on_click=generate_next_point,
            type="primary",
        )
