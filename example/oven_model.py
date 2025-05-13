from typing import Any, Dict, List

import numpy as np
import pandas as pd


def get_oven_model_label(
    *, temperature: int = 180, time: int = 10, mode: str = "Umluft", englisch_language: bool = False
):
    time_lower_bound = 2
    time_upper_bound = 20
    time_intervall = np.array([time_lower_bound, time_upper_bound])
    temp_lower_bound = 50
    temp_upper_bound = 250
    temp_intervall = np.array([temp_lower_bound, temp_upper_bound])

    mode = mode.lower()  # Konvertiere den Input in Kleinbuchstaben

    if "umluft" in mode or "heißluft" in mode or "hot air" in mode or "circulating" in mode:
        offset = 0
    elif "grill" in mode:
        offset = -15
    elif "ober" in mode and "unter" in mode or "upper" in mode and "lower" in mode:
        offset = 20
    else:
        return "Modus ist nicht bekannt. Wähle eins von Heißluft, Grill, oder Ober- und Unterhitze."

    if (
        temperature < np.min(temp_intervall)
        or temperature > np.max(temp_intervall)
        or time < np.min(time_intervall)
        or time > np.max(time_intervall)
    ):
        return (
            f"Eingabe befindet sich außerhalb der Grenzen:\n"
            f"Zeit {time_lower_bound} bis {time_upper_bound} Minuten\n"
            f"Temperatur: {temp_lower_bound} bis {temp_upper_bound} Grad Celsius."
        )

    c = 105 + offset
    alpha = 0.150
    beta = 245.0

    c2 = 60 + offset
    alpha2 = 0.060
    beta2 = 250.0

    f_x = beta * np.exp(-alpha * time) + c
    h_x = beta2 * np.exp(-alpha2 * time) + c2

    if temperature < f_x:
        label = "too light" if englisch_language else "zu hell"
    elif temperature > h_x:
        label = "too dark" if englisch_language else "zu dunkel"
    else:
        label = "optimal"

    return label


def convert_label_to_dict(label: str, englisch_language: bool = False) -> Dict[str, int]:
    if englisch_language:
        return {
            "too light": 0 if label != "too light" else 100,
            "optimal": 0 if label != "optimal" else 100,
            "too dark": 0 if label != "too dark" else 100,
        }
    return {
        "zu hell": 0 if label != "zu hell" else 100,
        "optimal": 0 if label != "optimal" else 100,
        "zu dunkel": 0 if label != "zu dunkel" else 100,
    }


def get_matching_column(df, keyword):
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    raise ValueError(f"Keine passende Spalte für '{keyword}' gefunden")


def get_complete_label_dict(
    df: pd.DataFrame, sub_part_label_keys: List[Any]
) -> Dict[str, Dict[int, Any]]:
    label_dict: Dict[str, Dict[int, Any]] = {}
    for key in sub_part_label_keys:
        if "bräune" in key or "Browning" in key:
            result = {}
            englisch_language = True if "Browning" in key else False
            for index in range(len(df)):
                if englisch_language:
                    temp_col = get_matching_column(df, "Temperature")
                    time_col = get_matching_column(df, "Time")
                    mode_col = get_matching_column(df, "Oven mode")
                else:
                    temp_col = get_matching_column(df, "Temperatur")
                    time_col = get_matching_column(df, "Zeit")
                    mode_col = get_matching_column(df, "Ofeneinstellung")

                label = get_oven_model_label(
                    temperature=df[temp_col][index],
                    time=df[time_col][index],
                    mode=df[mode_col][index],
                    englisch_language=englisch_language
                )

                result[index] = convert_label_to_dict(label, englisch_language=englisch_language)
            label_dict[key] = result
        else:
            label_dict[key] = {}

    return label_dict
