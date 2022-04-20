# -*- coding: utf-8 -*-
import re
from typing import Optional

from ..mrd import MrdDoc, MrdTextLikeObj
from .base_rule import DocRule, DocTextRule, LineTextRule, SpanRule
from .base_source import ZL, ZL_A, ZL_B, ZL_C, Source


class ZLRule1(DocRule):
    def apply(self, doc: MrdDoc) -> Optional[Source]:
        if "登录智联查看" in doc[-1][-1].text:
            return ZL_A


class ZLRule2(SpanRule):
    def apply(self, span: MrdTextLikeObj) -> Optional[Source]:
        if span.text == "我要联系TA":
            return ZL_A

        # white font color
        if span.text == "暂不合适" and min(span.color) >= 230:
            return ZL_A


class ZLRule3(LineTextRule):
    def __init__(self):
        self.pattern = re.compile(r"ID:.*\|\s*最近活跃时间:")

    def apply(self, line_text: str) -> Optional[Source]:
        if self.pattern.search(line_text):
            return ZL_B


# class ZLRule4(LineTextRule):
#     def __init__(self):
#         self.patterns = [
#             re.compile(r"现居住于.*\|.*\|.*[戸户]口"),
#             re.compile(r"现居住地:.*\|\s*[户戸]口:"),
#         ]
#
#     def apply(self, line_text: str) -> Optional[Source]:
#         for pattern in self.patterns:
#             if pattern.search(line_text):
#                 return ZL


class ZLRule5(DocTextRule):
    def __init__(self):
        self.pattern1 = re.compile(r"现居住于.*\|.*\|.*[戸户]口")
        self.pattern2 = re.compile(r"现居住地:.*\|\s*[户戸]口:")
        self.pattern3 = re.compile(r"\|.*(\d+-\d+\s*元/月|保密|\d+\s*元/月\s*以下)")

    def apply(self, doc_text: str) -> Optional[Source]:
        if (
            self.pattern1.search(doc_text) or self.pattern2.search(doc_text)
        ) and self.pattern3.search(doc_text):
            return ZL


class ZLRule6(DocRule):
    def apply(self, doc: MrdDoc) -> Optional[Source]:
        if "zhaopin.com" in doc[-1].to_text():
            return ZL_B


class ZLRule7(DocTextRule):
    dep = ZL

    def __init__(self):
        gender_pattern = r"(男|女)"
        age_pattern = r"\d{2}岁"
        birth_pattern = r"\(?\d{4}年\d{1,2}月\)?"
        work_experience_pattern = r"(应届毕业生|\d{1,2}.*经验|无工作经验|暂无经验)"

        self.pattern = re.compile(
            rf"{gender_pattern}{age_pattern}{birth_pattern}{work_experience_pattern}"
        )

    def apply(self, doc_text: str) -> Optional[Source]:
        if self.pattern.search(doc_text):
            return ZL_B


class ZLRule8(DocTextRule):
    def __init__(self):
        year_pattern = r"\d{4}"
        month_pattern = r"(1[012]|0[1-9])"
        time_range_pattern = rf"{year_pattern}\.{month_pattern}\s*-\s*({year_pattern}\.{month_pattern}|至今)"

        self.pattern = re.compile(
            rf"{time_range_pattern}.*\|.*(\d+-\d+\s*元/月|保密|\d+\s*元/月\s*以下)"
        )

    def apply(self, doc_text: str) -> Optional[Source]:
        if "简历更新时间" in doc_text and self.pattern.search(doc_text):
            return ZL_C


class ZLRule9(DocTextRule):
    dep = ZL

    def apply(self, doc_text: str) -> Optional[Source]:
        if "如需联系候选人" in doc_text:
            return ZL_A


class ZLRule10(DocTextRule):
    dep = ZL

    def __init__(self):
        gender_pattern = r"(男|女)"
        work_experience_pattern = r"(应届毕业生|\d{1,2}.*经验|无工作经验|暂无经验)"
        birth_pattern = r"\(?\d{4}年\d{1,2}月\)?"

        self.pattern = re.compile(
            rf"{gender_pattern}\|{work_experience_pattern}\|{birth_pattern}"
        )

    def apply(self, doc_text: str) -> Optional[Source]:
        if self.pattern.search(doc_text):
            return ZL_A


zl_rules = [
    ZLRule1(),
    ZLRule2(),
    ZLRule3(),
    ZLRule5(),
    ZLRule6(),
    ZLRule7(),
    ZLRule8(),
    ZLRule9(),
    ZLRule10(),
]
