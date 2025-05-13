from typing import cast, Dict, List

from experimental_design.data_parts import PartContainer
from experimental_design.dataset.labeling import LabelContainer


__all__ = [
    "PartDesignHandler",
]


class PartDesignHandler:
    def __init__(
        self,
        *,
        part_design_container: PartContainer,
        settings: PartContainer,
    ) -> None:
        self.settings = settings
        self.user_label_container = LabelContainer(
            label_settings_df=settings.collect_samples()
        )

        self.part_design_container = part_design_container
        self.part_design_container.generate(
            num_samples=self.user_label_container.num_samples
        )

        self.initial_df = self.part_design_container.collect_samples()
        self.optimised_df = self.initial_df.copy()

        self.categorical_columns = self.get_columns_by_string("categorical")
        self.categorical_factors = self.extract_categorical_factors()

        self.numerical_columns = self.get_columns_by_string("numerical")

        self.meta_columns = list(
            cast(
                PartContainer, self.part_design_container.sub_parts["meta"]
            ).sub_parts.keys()
        )
        self.experimental_design_columns = list(
            cast(
                PartContainer,
                self.part_design_container.sub_parts["experimental_design"],
            ).sub_parts
        )

    def get_columns_by_string(self, string_part: str) -> List[str]:
        return [col for col in self.optimised_df.columns if string_part in col]

    def extract_categorical_factors(self) -> Dict[str, List[str]]:
        categorical_factors = {}
        for categorical_column in self.categorical_columns:
            categorical_factors[categorical_column] = self.optimised_df[
                categorical_column
            ].unique()
        return categorical_factors
