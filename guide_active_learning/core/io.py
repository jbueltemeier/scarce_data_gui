import os
import pickle
import warnings
from typing import Any, cast, List

import matplotlib.pyplot as plt
import pandas as pd
from dot2tex import dot2tex

from guide_active_learning.misc import (
    get_default_dataset_path,
    get_default_model_path,
    get_default_path,
    get_default_result_path,
    get_default_train_path,
    make_folder_path,
    make_output_filename,
)

__all__ = [
    "get_default_folder_by_name",
    "read_csv_as_df",
    "read_excel_as_df",
    "load_dataset",
    "find_files",
    "save_pdf",
    "save_pickle",
    "load_pickle",
    "save_latex",
    "save_to_excel",
    "save_data_after_iteration",
    "save_active_learning_pickle",
    "save_plot_figure",
]


def get_default_folder_by_name(foldername: str) -> str:
    folder_functions = {
        "train": get_default_train_path,
        "result": get_default_result_path,
        "model": get_default_model_path,
        "dataset": get_default_dataset_path,
    }

    if foldername in folder_functions:
        return folder_functions[foldername]()
    else:
        warnings.warn(
            f"Specified folder name {foldername} is not a default. "
            f"The default Data folder is used.",
            UserWarning,
        )
        return get_default_path()


def read_csv_as_df(filename: str, foldername: str = "dataset") -> pd.DataFrame:
    load_path = make_folder_path(
        get_default_folder_by_name(foldername=foldername), filename=filename
    )
    return pd.read_csv(load_path)


def read_excel_as_df(
    *folders: str, filename: str, index_col: int = 0, foldername: str = "train"
) -> pd.DataFrame:
    load_path = make_folder_path(
        get_default_folder_by_name(foldername=foldername), *folders, filename=filename
    )
    return pd.read_excel(load_path, index_col=index_col)


def load_dataset(name: str) -> pd.DataFrame:
    dataset_path = make_folder_path(
        get_default_folder_by_name(foldername="dataset"), filename="datasets.pickle"
    )
    with open(dataset_path, "rb") as handle_dict:
        datasets = pickle.load(handle_dict)

    dataset = datasets.get(name)
    dataset.index.set_names("Obs", inplace=True)
    return dataset


def save_pdf(*folders: str, filename: str, foldername: str = "result") -> None:
    save_path = make_folder_path(
        get_default_folder_by_name(foldername=foldername), *folders, filename=filename
    )
    plt.savefig(
        save_path,
        dpi=1000,
        format="pdf",
        bbox_inches="tight",
    )


def save_pickle(
    *folders: str, results: Any, filename: str, foldername: str = "model"
) -> None:
    save_path = make_folder_path(
        get_default_folder_by_name(foldername=foldername), *folders, filename=filename
    )
    # save_path = "Data/result/"
    with open(save_path, "wb") as handle:
        pickle.dump(results, handle)


def load_pickle(*folders: str, filename: str, foldername: str = "model") -> List[Any]:
    load_path_file = make_folder_path(
        get_default_folder_by_name(foldername=foldername), *folders, filename=filename
    )
    with open(load_path_file, "rb") as handle:
        result = pickle.load(handle)

    return cast(List[Any], result)


def find_files(
    *folders: str, search_str: str, foldername: str = "train", bm_name: str = ""
) -> List[str]:
    search_path = make_folder_path(
        get_default_folder_by_name(foldername=foldername), bm_name, *folders
    )
    filenames = []
    for root, dirs, files in os.walk(search_path):
        for filename in files:
            if search_str in filename:
                filenames.append(os.path.join(root, filename))

    return filenames


def save_latex(
    *folders: str,
    text: str,
    filename: str,
    correct_output: bool = False,
    foldername: str = "result",
) -> None:
    latex_code = dot2tex(text, format="tikz", crop=True)
    if correct_output:
        latex_code = latex_code[
            latex_code.find("% Start of") : latex_code.find("End of")
        ]
        latex_code = latex_code.replace("latex'", "stealth")
        latex_code = latex_code.replace("<=", r"$\;\leq\;$")

    save_path = make_folder_path(
        get_default_folder_by_name(foldername=foldername), *folders, filename=filename
    )
    with open(save_path, "w") as handle:
        handle.write(latex_code)


def save_to_excel(
    *folders: str, data: pd.DataFrame, filename: str, foldername: str = "train"
) -> None:
    save_path = make_folder_path(
        get_default_folder_by_name(foldername=foldername),
        *folders,
        filename=filename,
    )
    data.to_excel(save_path)


def save_data_after_iteration(
    *folders: str, iteration: int, train_data: pd.DataFrame, pool_data: pd.DataFrame
) -> None:
    train_filename = make_output_filename(
        "train_pool_after_it", str(iteration), extension=".xlsx"
    )
    pool_filename = make_output_filename(
        "unl_pool_after_it", str(iteration), extension=".xlsx"
    )
    save_to_excel(*folders, filename=train_filename, data=train_data)
    save_to_excel(*folders, filename=pool_filename, data=pool_data)


def save_active_learning_pickle(
    *folders: str, filename: str, data: Any, foldername: str = "model"
) -> None:
    save_path_name = make_folder_path(
        get_default_folder_by_name(foldername=foldername), *folders, filename=filename
    )
    with open(save_path_name, "wb") as handle:
        pickle.dump(data, handle)


def save_plot_figure(
    *folders: str, filename: str, fig: plt.Figure, foldername: str = "result"
) -> None:
    save_path_name = make_folder_path(
        get_default_folder_by_name(foldername=foldername), *folders, filename=filename
    )
    fig.savefig(save_path_name)
    plt.close(fig)
