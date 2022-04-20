# -*- coding: utf-8 -*-
import re
from typing import Optional

from .base_rule import LineTextRule
from .base_source import LG, LG_A, Source


class LGRule1(LineTextRule):
    def __init__(self):
        self.pattern1 = re.compile(r"^\d{1,2}年工作经验/.{2,6}/\d{2}岁")
        self.pattern2 = re.compile(r"^\d{1,2}年工作经验\|.{2,6}\|\d{2}岁")

    def apply(self, line_text: str) -> Optional[Source]:
        if self.pattern1.search(line_text) or self.pattern2.search(line_text):
            return LG


class LGRule2(LineTextRule):
    dep = LG

    def apply(self, line_text: str) -> Optional[Source]:
        if line_text == "自我描述" or line_text == "个人优势":
            return LG_A


lg_rules = [LGRule1(), LGRule2()]
