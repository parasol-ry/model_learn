# -*- coding: utf-8 -*-
import re
from typing import Optional

from .base_rule import DocTextRule, LineTextRule
from .base_source import LP_A, LP_B, LP_C, Source


class LPRule1(DocTextRule):
    def apply(self, doc_text: str) -> Optional[Source]:
        if "精英简历" in doc_text and "简历编号" in doc_text:
            return LP_B


class LPRule2(DocTextRule):
    def __init__(self):
        self.pattern = re.compile(
            r"简历编号:.*所在地区:.*所在部门:.*汇报对象:.*下属人数:.*薪酬情况:.*", re.DOTALL
        )

    def apply(self, doc_text: str) -> Optional[Source]:
        if self.pattern.search(doc_text):
            return LP_C


class LPRule3(LineTextRule):
    def __init__(self):
        self.pattern = re.compile(r"^教育程度:.*职业状态:.*")

    def apply(self, line_text: str) -> Optional[Source]:
        if self.pattern.search(line_text):
            return LP_A


lp_rules = [LPRule1(), LPRule2(), LPRule3()]
