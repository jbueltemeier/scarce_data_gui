from typing import TYPE_CHECKING

import streamlit as st

from experimental_design.visualisation import plot_function_from_df

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

__all__ = [
    "display_results",
]


def display_results(col: "DeltaGenerator", active_learning_mode: bool = False) -> None:
    with col:
        df_meta = st.session_state["dataset"].get_data(
            user_name=st.session_state["name"],
            meta=True,
            active_learning_mode=active_learning_mode,
        )
        df_result_settings = st.session_state["dataset"].get_data(
            user_name=st.session_state["name"],
            meta=False,
            active_learning_mode=active_learning_mode,
        )

        df_result_function = df_result_settings[
            df_result_settings.index.str.contains("_function", na=False)
        ]

        df_result_settings = df_result_settings[
            ~df_result_settings.index.str.contains("_function", na=False)
        ]

        if df_result_function.empty:
            df_result_function = None

        col1, col2, _ = st.columns([0.5, 0.3, 0.1])
        with col1:
            if df_result_function is not None:
                st.plotly_chart(
                    st.session_state["dataset"].plot_input_space(
                        user_name=st.session_state["name"],
                        active_learning_mode=active_learning_mode,
                    )
                )

                df_result_function = df_result_function.to_frame().T
                df_result_function.columns.name = None
                st.pyplot(plot_function_from_df(df_result_function.iloc[[0]]))
            else:
                st.plotly_chart(
                    st.session_state["dataset"].plot_input_space(
                        user_name=st.session_state["name"],
                        active_learning_mode=active_learning_mode,
                    )
                )
        with col2:
            st.header("Bericht")
            st.subheader("Allgemein")
            if df_meta is not None:
                st.table(df_meta.astype(str).transpose())
            st.subheader("Einstellung")
            if df_result_settings is not None:
                st.table(df_result_settings.astype(str).transpose())
