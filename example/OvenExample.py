import os
import time

import streamlit as st
from example.oven_model import get_oven_model_label
from scarce_data_gui.misc import get_default_example_path


def main():
    st.title("Oven Model App")

    input_temperature = st.number_input(
        "Temperatur (50-250°C)", min_value=50.0, step=0.01
    )
    input_time = st.number_input("Zeit (2-20 Minuten)", min_value=2.0, step=0.01)

    options = ["Heißluft", "Grill", "Ober- und Unterhitze"]
    selected_option = st.selectbox("Wählen Sie eine Ofeneinstellung:", options)

    timer_input = st.checkbox("Timer", value=True)
    if st.button("Brötchen backen"):
        result = get_oven_model_label(
            temperature=input_temperature, time=input_time, mode=selected_option
        )
        if result == "zu hell":
            sleep_time = 2
            result_image = "images/hell.png"
        elif result == "optimal":
            sleep_time = 4
            result_image = "images/optimal.png"
        elif result == "zu dunkel":
            sleep_time = 6
            result_image = "images/dunkel.png"
        else:
            sleep_time = None

        if sleep_time is not None:
            if timer_input:
                with st.spinner("Brötchen wird gebacken..."):
                    time.sleep(sleep_time)
                st.subheader(f"Frisch gebackenes Brötchen. Ergebnis: {result} ")
                st.image(os.path.join(get_default_example_path(), result_image))
            else:
                st.subheader(f"{result} ")
        else:
            st.subheader(f"{result}")


if __name__ == "__main__":
    main()
