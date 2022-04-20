# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "Source",
    "LP",
    "ZL",
    "QCWY",
    "LG",
    "OTHER",
    "LP_A",
    "LP_B",
    "LP_C",
    "ZL_A",
    "ZL_B",
    "ZL_C",
    "LG_A",
    "QCWY_A",
    "QCWY_B",
    "QCWY_C",
    "QCWY_D",
    "QCWY_E",
    "ALL_SOURCE",
]


@dataclass
class Source:
    name: str
    parent: Optional["Source"]
    is_leaf: bool

    def __hash__(self) -> int:
        return hash(self.name)


LP = Source("LP", None, False)
ZL = Source("ZL", None, False)
LG = Source("LG", None, False)
QCWY = Source("QCWY", None, False)

LP_A = Source("LP_A", LP, True)
LP_B = Source("LP_B", LP, True)
LP_C = Source("LP_C", LP, True)

ZL_A = Source("ZL_A", ZL, True)
ZL_B = Source("ZL_B", ZL, True)
ZL_C = Source("ZL_C", ZL, True)

LG_A = Source("LG_A", LG, True)

QCWY_A = Source("QCWY_A", QCWY, True)
QCWY_B = Source("QCWY_B", QCWY, True)
QCWY_C = Source("QCWY_C", QCWY, True)
QCWY_D = Source("QCWY_D", QCWY, True)
QCWY_E = Source("QCWY_E", QCWY, True)

OTHER = Source("OTHER", None, True)

ALL_SOURCE = [
    LP,
    ZL,
    LG,
    QCWY,
    LP_A,
    LP_B,
    LP_C,
    ZL_A,
    ZL_B,
    ZL_C,
    LG_A,
    QCWY_A,
    QCWY_B,
    QCWY_C,
    QCWY_D,
    QCWY_E,
    OTHER,
]
