from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, cast, Dict, Optional, Sized, Union

import matplotlib
import numpy as np
import pandas as pd

from experimental_design.core import range_with_float
from experimental_design.data_parts import (
    FactorFunctionNumberInstancePart,
    FunctionNumberInstancePart,
    FunctionTimeInstancePart,
    NumberInstancePart,
    StaticStringInstancePart,
    StringNumberInstancePart,
)

from experimental_design.data_parts.data_field import DataFieldBase, info

from experimental_design.visualisation import (
    function_visualisation,
    visualize_centrifuge_test,
)

from .instance_part import _InstancePartBase

__all__ = [
    "_PartContainer",
    "PartContainer",
    "MetaPart",
    "_DataFieldContainerBase",
    "_FunctionPartContainerBase",
    "ExponentialFunctionPart",
    "GaussFunctionPart",
    "_CustomPartBase",
]


class _PartContainer(ABC):
    def __init__(self) -> None:
        self.sub_parts: Dict[str, Union[_InstancePartBase, _PartContainer]] = {}

    def generate(self, num_samples: int) -> None:
        for param in self.sub_parts.values():
            param.generate(num_samples=num_samples)

    def get_samples_from_part(
        self, param_str: str, design_space: bool = True
    ) -> pd.DataFrame:
        if param_str not in self.sub_parts.keys():
            msg = f"Attribute {param_str} is not included in this part."
            raise AttributeError(msg)

        if isinstance(self.sub_parts[param_str], _PartContainer):
            if not design_space:
                if isinstance(self.sub_parts[param_str], _DataFieldContainerBase):
                    return cast(
                        _FunctionPartContainerBase, self.sub_parts[param_str]
                    ).translate()
            return cast(_PartContainer, self.sub_parts[param_str]).collect_samples(
                design_space=design_space
            )
        else:
            return cast(_InstancePartBase, self.sub_parts[param_str]).translate(
                design_space=design_space
            )

    def collect_samples(self, design_space: bool = True) -> pd.DataFrame:
        dataframes = []
        for key, sub_part in self.sub_parts.items():
            dataframes.append(
                self.get_samples_from_part(key, design_space=design_space)
            )

        return pd.concat(dataframes, axis=1)

    def assign_samples_to_part(
        self,
        df: pd.DataFrame,
        sub_part: Union[_InstancePartBase, "_PartContainer"],
    ) -> Union[_InstancePartBase, "_PartContainer"]:
        if isinstance(sub_part, _PartContainer):
            self._assign_to_sub_parts(df, sub_part.sub_parts)
        elif isinstance(sub_part, _InstancePartBase):
            sub_part.samples = df
        else:
            raise NotImplementedError(f"not implemented {type(sub_part)}")
        return sub_part

    def _assign_to_sub_parts(
        self,
        df: pd.DataFrame,
        sub_parts: Dict[str, Union[_InstancePartBase, "_PartContainer"]],
    ) -> None:
        pos = 0
        for key, part in sub_parts.items():
            pos_end = pos + len(cast(Sized, part))
            sub_parts[key] = self.assign_samples_to_part(df.iloc[:, pos:pos_end], part)
            pos = pos_end

    def assign_samples(self, df: pd.DataFrame) -> None:
        self._assign_to_sub_parts(df, self.sub_parts)

    def __len__(self) -> int:
        return sum(len(cast(Sized, value)) for value in self.sub_parts.values())


class PartContainer(_PartContainer):
    def __init__(
        self,
        *,
        parts: Dict[str, Union[_InstancePartBase, _PartContainer]],
    ) -> None:
        super().__init__()
        self.sub_parts = parts


class MetaPart(PartContainer):
    def __init__(
        self,
        *,
        parts: Dict[str, Union[_InstancePartBase, _PartContainer]],
    ) -> None:
        meta_parts: Dict[str, Union[_InstancePartBase, _PartContainer]] = {
            "Berichtsnummer_static": StaticStringInstancePart(
                fields={"name": "Berichtsnummer", "sample_value": "augmented"}
            ),
            "Datum_static": StaticStringInstancePart(
                fields={
                    "name": "Datum",
                    "sample_value": datetime.now().strftime("%Y-%m-%d"),
                }
            ),
            **parts,
        }
        super().__init__(parts=meta_parts)


class _DataFieldContainerBase(PartContainer, DataFieldBase):
    own_fields: Dict[str, Any] = {}

    @classmethod
    def get_own_fields(cls) -> Dict[str, Any]:
        return cls.own_fields

    @abstractmethod
    def translate(self) -> pd.DataFrame:
        pass

    @classmethod
    @abstractmethod
    def info(cls, fields: Optional[Dict[str, Any]] = None) -> None:
        pass

    def __repr__(self) -> str:
        instance_html = """<div style="display: flex; gap: 10px;">"""
        for instance in self.sub_parts.values():
            instance_html += f"<div>{instance}</div>"
        instance_html += "</div>"
        return instance_html


class _FunctionPartContainerBase(_DataFieldContainerBase):
    own_fields: Dict[str, Any] = {}

    def __init__(
        self,
        fields: Dict[str, Any],
        parts: Dict[str, Union[_InstancePartBase, _PartContainer]],
    ) -> None:
        self._instance_suffix = "_function"
        name = fields.get("name")
        DataFieldBase.__init__(self, f"{name}{self._instance_suffix}", fields)
        PartContainer.__init__(self, parts=parts)

    def translate(self) -> pd.DataFrame:
        x_times = self.get_times()
        function_values = self.calculate_function(x=x_times)
        column_names = [str(time) + "_function" for time in x_times]
        return pd.DataFrame(function_values, columns=column_names)

    def get_times(self) -> np.ndarray:
        return np.array(
            cast(FunctionTimeInstancePart, self.sub_parts["time"]).times, dtype=float
        )

    @abstractmethod
    def calculate_function(
        self, x: np.ndarray, fields: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def function_str() -> str:
        pass

    @classmethod
    def info(
        cls,
        fields: Optional[Dict[str, Any]] = None,
        start_value: int = 0,
        stop_value: int = 60,
        step_size: float = 0.1,
    ) -> matplotlib.figure.Figure:
        float_range_list = list(
            range_with_float(start=start_value, stop=stop_value, step=step_size)
        )
        float_range_array = np.array(float_range_list)
        result_array = np.expand_dims(float_range_array, axis=1)
        if fields is not None:
            return function_visualisation(
                x=result_array,
                y=cls.calculate_function(
                    cls, x=result_array, fields=fields  # type: ignore
                ),
            )
        else:
            return function_visualisation(
                x=result_array, y=cls.calculate_function(x=result_array, fields=fields)  # type: ignore
            )

    def __repr__(self) -> str:
        instance_html = f""" <body><h5>{self.__class__.__name__}</h5>
            <p><strong>Name:</strong> {self.instance_name}</p>
        """
        instance_html += super().__repr__()
        instance_html += "</body>"
        return instance_html


class ExponentialFunctionPart(_FunctionPartContainerBase):
    own_fields = {
        "name": {"type": str, "default": None, "label": "**Exponential Funktion**"},
        "a": {
            "type": FunctionNumberInstancePart,
            "default": None,
            "label": "Parameter a",
        },
        "b": {
            "type": FactorFunctionNumberInstancePart,
            "default": None,
            "label": "Parameter b",
        },
        "c": {
            "type": FactorFunctionNumberInstancePart,
            "default": None,
            "label": "Parameter c",
        },
        "d": {
            "type": FunctionNumberInstancePart,
            "default": None,
            "label": "Parameter d",
        },
        "time": {
            "type": FunctionTimeInstancePart,
            "default": None,
            "label": "Zeitpunkte",
        },
        "info": {"type": info, "default": False, "label": "Information Show"},
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        parts = {
            "a": FunctionNumberInstancePart(
                fields=cast(Dict[str, Any], fields.get("a"))
            ),
            "b": FactorFunctionNumberInstancePart(
                fields=cast(Dict[str, Any], fields.get("b"))
            ),
            "c": FactorFunctionNumberInstancePart(
                fields=cast(Dict[str, Any], fields.get("c"))
            ),
            "d": FunctionNumberInstancePart(
                fields=cast(Dict[str, Any], fields.get("d"))
            ),
            "time": FunctionTimeInstancePart(
                fields=cast(Dict[str, Any], fields.get("time"))
            ),
        }
        super().__init__(fields, parts=cast(Dict[str, Any], parts))

    @staticmethod
    def function_str() -> str:
        return "$a \\cdot e^{ -b \\cdot x } - (a + b) \\cdot e^{ -(b+c) \\cdot x } + d$"

    def calculate_function(
        self, x: np.ndarray, fields: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        # a * exp(-b * x) - (a + b) * exp(-(b+c) * x) + d
        def get_param_values(
            param_key: str, num_function_samples: int = 10
        ) -> np.ndarray:
            if fields:
                return np.random.uniform(
                    fields[param_key]["min_value"],
                    fields[param_key]["max_value"],
                    num_function_samples,
                ).squeeze()
            return cast(
                np.ndarray,
                self.get_samples_from_part(param_key, design_space=False).to_numpy(),
            )

        a, b, c, d = (
            get_param_values(param_key=param) for param in ["a", "b", "c", "d"]
        )
        return cast(np.ndarray, a * np.exp(-b * x) - (a + d) * np.exp(-(b + c) * x) + d)


class GaussFunctionPart(_FunctionPartContainerBase):
    own_fields = {
        "name": {"type": str, "default": None, "label": "**Gauss Funktion**"},
        "a": {
            "type": FunctionNumberInstancePart,
            "default": None,
            "label": "Parameter a",
        },
        "b": {
            "type": FunctionNumberInstancePart,
            "default": None,
            "label": "Parameter b",
        },
        "c": {
            "type": FunctionNumberInstancePart,
            "default": None,
            "label": "Parameter c",
        },
        "time": {
            "type": FunctionTimeInstancePart,
            "default": None,
            "label": "Zeitpunkte",
        },
        "info": {"type": info, "default": False, "label": "Information Show"},
    }

    def __init__(self, fields: Dict[str, Any]) -> None:
        parts: Dict[str, Union[_InstancePartBase, _PartContainer]] = {
            "a": FunctionNumberInstancePart(
                fields=cast(Dict[str, Any], fields.get("a"))
            ),
            "b": FunctionNumberInstancePart(
                fields=cast(Dict[str, Any], fields.get("b"))
            ),
            "c": FunctionNumberInstancePart(
                fields=cast(Dict[str, Any], fields.get("c"))
            ),
            "time": FunctionTimeInstancePart(
                fields=cast(Dict[str, Any], fields.get("time"))
            ),
        }
        super().__init__(fields, parts=parts)

    @staticmethod
    def function_str() -> str:
        return "$a \\cdot e^{-(x - b)^2) / (2 \\cdot c^2)}$"

    def calculate_function(
        self, x: np.ndarray, fields: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        # a * exp(-(x - b)^2/ (2 * c^2))
        def get_param_values(
            param_key: str, num_function_samples: int = 10
        ) -> np.ndarray:
            if fields:
                return np.random.uniform(
                    fields[param_key]["min_value"],
                    fields[param_key]["max_value"],
                    num_function_samples,
                ).squeeze()
            return cast(
                np.ndarray,
                self.get_samples_from_part(param_key, design_space=False).to_numpy(),
            )

        a, b, c = (get_param_values(param_key=param) for param in ["a", "b", "c"])
        return cast(
            np.ndarray, a * np.exp(-np.divide(np.square(x - b), 2 * np.square(c)))
        )


class _CustomPartBase(_DataFieldContainerBase):
    own_fields: Dict[str, Any] = {}

    def __init__(
        self,
        fields: Dict[str, Any],
    ) -> None:
        self._instance_suffix = "_custom"
        name = fields.get("name")
        DataFieldBase.__init__(self, f"{name}{self._instance_suffix}", fields)

        parts = self.create_instances(fields)
        PartContainer.__init__(self, parts=parts)

    def create_instances(
        self, params_dict: Dict[str, Any]
    ) -> Dict[str, Union[_InstancePartBase, _PartContainer]]:
        instances = {}

        for key, meta in self.own_fields.items():
            cls_type = meta["type"]
            if cls_type in (str, int, float, bool, info):
                continue
            param_data = params_dict.get(key)

            if isinstance(cls_type, type):
                if param_data is not None:
                    instances[key] = cls_type(param_data)
                else:
                    instances[key] = cls_type()
            elif callable(cls_type):
                if param_data is not None:
                    instances[key] = cls_type(param_data)
                else:
                    instances[key] = cls_type(None)
            else:
                raise TypeError(f"Unbekannter Typ für Schlüssel {key}: {cls_type}")

        return instances

    @abstractmethod
    def translate(self) -> pd.DataFrame:
        pass

    @classmethod
    @abstractmethod
    def info(cls, fields: Optional[Dict[str, Any]] = None) -> matplotlib.figure.Figure:
        pass


class CentrifugeVolumenPart(_CustomPartBase):
    own_fields = {
        "name": {"type": str, "default": None, "label": "**Volumen Instanz**"},
        "info": {"type": info, "default": False, "label": "Information Show"},
        "tMax": {
            "type": NumberInstancePart,
            "default": None,
            "label": "Maximale Auswertungszeit",
        },
        "function": {
            "type": ExponentialFunctionPart,
            "default": None,
            "label": "Funktion",
        },
        "volHeavyPhase": {
            "type": NumberInstancePart,
            "default": None,
            "label": "Volumen Prozent Schwere Phase",
        },
        "consSolids": {
            "type": StringNumberInstancePart,
            "default": None,
            "label": "Konsistenz Feststoffe",
        },
        "turbLightPhase": {
            "type": StringNumberInstancePart,
            "default": None,
            "label": "Aussehen leichte Phase",
        },
        "turbHeavyPhase": {
            "type": StringNumberInstancePart,
            "default": None,
            "label": "Aussehen schwere Phase",
        },
        "abrasion": {
            "type": StringNumberInstancePart,
            "default": None,
            "label": "Abrasion",
        },
    }

    def translate(self) -> pd.DataFrame:
        volSolids = cast(
            _FunctionPartContainerBase, self.sub_parts["function"]
        ).translate()
        volHeavyPhase = cast(
            _InstancePartBase, self.sub_parts["volHeavyPhase"]
        ).translate(design_space=False)
        volLightPhase = 100 - volSolids
        volLightPhase = volLightPhase.subtract(
            volHeavyPhase["volHeavyPhase_numerical"], axis=0
        )
        result_dict: Dict[str, Any] = {}
        for column in volSolids.columns:
            result_dict[column] = []
            for i in range(len(volSolids)):
                result_dict[column].append(
                    {
                        "volSolids": volSolids[column][i],
                        "volLightPhase": volLightPhase[column][i],
                        "volHeavyPhase": volHeavyPhase["volHeavyPhase_numerical"][i],
                    }
                )
        df = pd.DataFrame(result_dict)
        tMax = cast(_InstancePartBase, self.sub_parts["tMax"]).translate(
            design_space=False
        )
        consSolids = cast(_InstancePartBase, self.sub_parts["consSolids"]).translate(
            design_space=False
        )
        turbLightPhase = cast(
            _InstancePartBase, self.sub_parts["turbLightPhase"]
        ).translate(design_space=False)
        turbHeavyPhase = cast(
            _InstancePartBase, self.sub_parts["turbHeavyPhase"]
        ).translate(design_space=False)
        abrasion = cast(_InstancePartBase, self.sub_parts["abrasion"]).translate(
            design_space=False
        )
        df = pd.concat(
            [tMax, df, consSolids, turbLightPhase, turbHeavyPhase, abrasion], axis=1
        )
        return df

    @classmethod
    def info(cls, fields: Optional[Dict[str, Any]] = None) -> matplotlib.figure.Figure:
        phases = ["Feststoffe", "Phase 1", "Phase 2"]
        percentages = [12, 48, 40]
        return visualize_centrifuge_test(phases, percentages)
