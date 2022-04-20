# -*- coding: utf-8 -*-

import re
from typing import Optional

from ..mrd import MrdTextLikeObj
from .base_rule import DocTextRule, SpanRule
from .base_source import QCWY, QCWY_A, QCWY_B, QCWY_C, QCWY_D, QCWY_E, Source


class QCWYRule1(SpanRule):
    def __init__(self):
        self.pattern = re.compile(r"^\(?ID:\d{5,10}\)?$")

    def apply(self, span: MrdTextLikeObj) -> Optional[Source]:

        if self.pattern.match(span.text):
            return QCWY


class QCWYRule2(DocTextRule):
    dep = QCWY

    def __init__(self):
        gender_pattern = r"(男|女)"
        work_experience_pattern = r"(应届毕业生|\d{1,2}.*经验|无工作经验|暂无经验)"
        age_birth_1_pattern = r"\d{2}岁\(\d{4}\/\d{1,2}\/\d{1,2}\)"
        age_birth_2_pattern = r"\d{2}岁\(\d{4}年\d{1,2}月\d{1,2}日\)"
        age_pattern = r"\d{2}岁"
        self.qcwy_a_pattern = re.compile(
            rf"{age_birth_1_pattern}\|{work_experience_pattern}"
        )
        self.qcwy_b_pattern = re.compile(
            rf"{work_experience_pattern}\|{gender_pattern}\|{age_birth_2_pattern}"
        )
        self.qcwy_c_pattern = re.compile(
            rf"{gender_pattern}\|{age_birth_2_pattern}\|现居住"
        )
        self.qcwy_d_pattern = re.compile(
            rf"{gender_pattern}\|{age_birth_1_pattern}\|现居住"
        )
        self.qcwy_e_pattern = re.compile(
            rf"{gender_pattern}\|{age_pattern}(\|现居住.*)?\|{work_experience_pattern}"
        )

    def apply(self, doc_text: str) -> Optional[Source]:
        if self.qcwy_a_pattern.search(doc_text):
            return QCWY_A
        elif self.qcwy_b_pattern.search(doc_text):
            return QCWY_B
        elif self.qcwy_c_pattern.search(doc_text):
            return QCWY_C
        elif self.qcwy_d_pattern.search(doc_text):
            return QCWY_D
        elif self.qcwy_e_pattern.search(doc_text):
            return QCWY_E


qcwy_rules = [QCWYRule1(), QCWYRule2()]
