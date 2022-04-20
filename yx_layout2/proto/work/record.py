# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Tuple

LineWorkTag = Tuple[int, int, int, str]
LineWorkTags = List[LineWorkTag]
Position = Tuple[float, float, float, float]
LinePositionTag = Tuple[int, int, Position]
LinePositionTags = List[Tuple[int, int, Position]]


@dataclass
class WorkRecord:
    # len(position_tags) == len(texts) == len(line_ids)

    resume_id: str
    line_ids: List[int]
    texts: List[str]
    tags: List[LineWorkTags]
    position_tags: List[LinePositionTags]
