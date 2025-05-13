from typing import Any, cast, Dict, List, Optional

import matplotlib

import pandas as pd

from experimental_design.data_parts import (
    _InstancePartBase,
    NumberInstancePart,
    StringNumberInstancePart,
)
from experimental_design.data_parts.container import (
    _CustomPartBase,
    _FunctionPartContainerBase,
    ExponentialFunctionPart,
)

from experimental_design.data_parts.data_field import info
from experimental_design.visualisation import visualize_centrifuge_test


__all__ = [
    "CentrifugeVolumenPart",
    "additional_data_parts",
]


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
            volHeavyPhase["Instanz_volHeavyPhase (in Volumenprozent)_numerical"], axis=0
        )
        result_dict: Dict[str, Any] = {}
        for column in volSolids.columns:
            result_dict[column] = []
            for i in range(len(volSolids)):
                result_dict[column].append(
                    {
                        "volSolids": volSolids[column][i],
                        "volLightPhase": volLightPhase[column][i],
                        "volHeavyPhase": volHeavyPhase[
                            "Instanz_volHeavyPhase (in Volumenprozent)_numerical"
                        ][i],
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


def additional_data_parts() -> List[Dict[str, Any]]:
    return [
        # {
        #     "class": CentrifugeVolumenPart,
        #     "name": "Volumen Phasen Auswertung",
        # },
    ]
