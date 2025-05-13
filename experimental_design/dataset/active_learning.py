from typing import List, Optional

import pandas as pd

from experimental_design.data_parts import PartContainer
from experimental_design.dataset.labeling import ActiveLearningLabelContainer

__all__ = [
    "ActiveLearning",
]


class ActiveLearning:
    def __init__(self) -> None:
        # TODO: load this from settings
        self.alpha_weight = 0.8
        self.ensemble_size = 10
        self.max_depth = 10
        self.min_info_gain = 0.075
        self.guide_split = "mixed"
        self.num_samples = 100
        self.active_learning_scenario = "qbc_exex_rf"
        self.iteration = 0

        self.columns_to_keep: Optional[List[str]] = None
        self.active_learning_label: Optional[str] = None

        self.active_learning_datapoints = pd.DataFrame()
        self.active_learning_pool = pd.DataFrame()

        self.active_learning_label_container: Optional[
            ActiveLearningLabelContainer
        ] = None

    def init_active_learning_pool(
        self,
        part_design_container: PartContainer,
        settings: PartContainer,
    ) -> None:
        part_design_container = part_design_container
        part_design_container.generate(num_samples=self.num_samples)
        self.active_learning_pool = part_design_container.collect_samples()
        self.active_learning_pool["target"] = None
        self.active_learning_label_container = ActiveLearningLabelContainer(
            label_settings_df=settings.collect_samples()
        )

    def update_active_learning_data(self, datapoint: pd.Series) -> None:
        if len(datapoint) - 1 < len(self.active_learning_pool.columns):
            index = datapoint.iloc[0]
            datapoint = self.active_learning_pool.iloc[index, :]
        else:
            index = datapoint["index"]
            datapoint = datapoint.drop(["index"])
        datapoint[
            "Berichtsnummer_static"
        ] = "active_learning"  # change default to show active learning
        self.active_learning_datapoints = pd.concat(
            [self.active_learning_datapoints, datapoint.to_frame().T], ignore_index=True
        )
        self.active_learning_pool = self.active_learning_pool.drop(index=index)

    def filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_keep is None:
            return df
        if "target" in df.columns:
            columns_to_keep = self.columns_to_keep + ["target"]
            return df[columns_to_keep]
        return df[self.columns_to_keep].copy()
