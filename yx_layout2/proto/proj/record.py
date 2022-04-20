# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Tuple

LineProjTag = Tuple[int, int, int, str]
LineProjTags = List[LineProjTag]
Position = Tuple[float, float, float, float]
LinePositionTag = Tuple[int, int, Position]
LinePositionTags = List[Tuple[int, int, Position]]


@dataclass
class ProjRecord:
    # len(position_tags) == len(texts) == len(line_ids)

    resume_id: str
    line_ids: List[int]
    texts: List[str]
    tags: List[LineProjTags]
    position_tags: List[LinePositionTags]
