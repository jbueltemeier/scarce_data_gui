from typing import Any, Dict

import numpy as np

from guide_active_learning.core import (
    calculate_mean_std,
    find_files,
    load_pickle,
    plot_benchmark_mean_std,
    save_pickle,
)
from guide_active_learning.misc import make_output_filename


__all__ = [
    "read_results",
    "extract_results",
    "analyse_dataset_results",
    "evaluate_results",
]


def read_results(*folders: str, filename: str) -> Dict[str, Any]:
    result = load_pickle(*folders, filename=filename)

    num_benchmark = len(result)
    num_steps_per_benchmark = len(result[0])
    result_arr = np.ones((num_benchmark, num_steps_per_benchmark))
    for benchmark in range(len(result)):
        result_arr[benchmark, :] = [i.get("score") for i in result[benchmark].values()]

    filename_parts = filename.split("\\")[-1]
    filename_parts = filename_parts.split("__")
    if len(filename_parts) == 8:
        result_dict = {
            "result_array": result_arr,
            "active_learning_method": filename_parts[2],
            "split_type": filename_parts[3],
            "num_datapoints": filename_parts[4],
            "alpha_weight": filename_parts[5],
            "ensemble_size": filename_parts[6],
            "min_info_gain": filename_parts[7],
        }
    elif len(filename_parts) == 7:
        result_dict = {
            "result_array": result_arr,
            "active_learning_method": filename_parts[2],
            "split_type": filename_parts[3],
            "num_datapoints": filename_parts[4],
            "ensemble_size": filename_parts[5],
            "min_info_gain": filename_parts[6],
        }
    else:
        raise Exception("Invalid filename")
    return result_dict


def extract_results(
    dataset_str: str,
    bm_name: str,
) -> None:
    results_files = find_files(
        f"{dataset_str}_results",
        search_str=dataset_str,
        foldername="result",
        bm_name=bm_name,
    )

    all_results = []
    for file in results_files:
        all_results.append(read_results(f"{dataset_str}_results", filename=file))

    filename = make_output_filename(
        "results", dataset_str, "extracted", extension=".pickle"
    )
    save_pickle(f"{dataset_str}_results", results=all_results, filename=filename)


def create_label(result_dict: Dict[str, str]) -> str:
    if len(result_dict) == 6:
        keys = [
            "active_learning_method",
            "split_type",
            "ensemble_size",
            "min_info_gain",
        ]
    elif len(result_dict) == 7:
        keys = [
            "active_learning_method",
            "split_type",
            "alpha_weight",
            "ensemble_size",
            "min_info_gain",
        ]
    parts = [result_dict[key] for key in keys]
    return "__".join([*parts])


def analyse_dataset_results(dataset_str: str, run_sizes: list) -> None:
    results_files = find_files(
        f"{dataset_str}_results", search_str=dataset_str, foldername="model"
    )
    result = load_pickle(f"{dataset_str}_results", filename=results_files[0])

    for num_datapoints in run_sizes:
        data = [
            (
                create_label(result_dict),
                *calculate_mean_std(result_dict["result_array"]),
            )
            for result_dict in result
            if str(num_datapoints) in result_dict["num_datapoints"]
        ]
        plot_benchmark_mean_std(
            data,
            dataset_str=dataset_str,
            num_datapoints=num_datapoints,
            save_figure=True,
        )


def evaluate_results(dataset_str: str, bm_name: str, run_sizes: list) -> None:
    extract_results(dataset_str, bm_name)
    analyse_dataset_results(dataset_str, run_sizes)
