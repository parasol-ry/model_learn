# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Protocol, Sequence

from ..mrd.core import MrdTextLine
from .record import LineWorkTag

MrdBlock = Sequence[MrdTextLine]


class Tagger(Protocol):
    def tag(self, block: MrdBlock) -> List[LineWorkTag]:
        ...


class ZhiLian1WorkBlockTagger:
    def __init__(self) -> None:
        re_year = r"\d{4}"
        re_month = "(1[012]|0[1-9])"
        self.time_com_pattern = re.compile(
            rf"""
                ^
                (?P<time_range>{re_year}(\.|/){re_month}\s*-\s*({re_year}(\.|/){re_month}|至今))
                \s*
                (?P<company>.*?)
                \s*
                (\(\d.*(年|月)\))
                $
                """,
            re.VERBOSE,
        )
        self.title_salary_pattern = re.compile(
            r"""
            ^
            (?P<title>.*?)
            \s*\|\s*
            (?P<salary>(\d+-\d+\s*元/月|保密|\d+\s*元/月\s*以下))
            $
            """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineWorkTag]:
        tags = []
        for mrd_text_line in mrd_block:
            line_text = mrd_text_line.to_text("")
            match_obj = self.time_com_pattern.match(line_text)
            if match_obj:
                time_tag = (
                    mrd_text_line.id,
                    match_obj.start("time_range"),
                    match_obj.end("time_range"),
                    "timeRange",
                )
                tags.append(time_tag)
                com_tag = (
                    mrd_text_line.id,
                    match_obj.start("company"),
                    match_obj.end("company"),
                    "company",
                )
                tags.append(com_tag)
            else:
                match_obj = self.title_salary_pattern.match(line_text)
                if match_obj:
                    time_tag = (
                        mrd_text_line.id,
                        match_obj.start("title"),
                        match_obj.end("title"),
                        "title",
                    )
                    tags.append(time_tag)
                    salary_tag = (
                        mrd_text_line.id,
                        match_obj.start("salary"),
                        match_obj.end("salary"),
                        "salary",
                    )
                    tags.append(salary_tag)
        return tags


class ZhiLian2WorkBlockTagger:
    def __init__(self) -> None:
        re_year = r"\d{4}"
        re_month = "(1[012]|0[1-9])"
        self.pattern = re.compile(
            rf"""
                ^
                (?P<time_range>{re_year}(\.|/){re_month}\s*-\s*({re_year}(\.|/){re_month}|至今))
                (\(\d.*(年|月)\))
                (?P<company>.*?)
                \|
                (?P<title>.*?)
                (?P<salary>(\d+-\d+\s*元/月|保密|\d+\s*元/月\s*以下))
                $
                """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineWorkTag]:
        tags = []
        for mrd_text_line in mrd_block:
            line_text = mrd_text_line.to_text("")
            match_obj = self.pattern.match(line_text)
            if match_obj:
                time_tag = (
                    mrd_text_line.id,
                    match_obj.start("time_range"),
                    match_obj.end("time_range"),
                    "timeRange",
                )
                tags.append(time_tag)
                com_tag = (
                    mrd_text_line.id,
                    match_obj.start("company"),
                    match_obj.end("company"),
                    "company",
                )
                tags.append(com_tag)
                title_tag = (
                    mrd_text_line.id,
                    match_obj.start("title"),
                    match_obj.end("title"),
                    "title",
                )
                tags.append(title_tag)
                salary_tag = (
                    mrd_text_line.id,
                    match_obj.start("salary"),
                    match_obj.end("salary"),
                    "salary",
                )
                tags.append(salary_tag)
        return tags


class QCWY1WorkBlockTagger:
    def __init__(self) -> None:
        re_year = r"\d{4}"
        re_month = "(1[012]|[1-9])"
        self.time_pattern = re.compile(
            rf"""
                ^
                (?P<time_range>{re_year}\s*/{re_month}\s*-{{1,2}}\s*({re_year}\s*/{re_month}|至今))
                \s*
                (?P<title>.*?)
                \s*
                (\|\s*(?P<depart>.*?))?
                $
                """,
            re.VERBOSE,
        )
        self.com_pattern = re.compile(
            r"""
                    ^
                    (?P<company>.*?)
                    \s*
                    (\[\s*\d+.*\])
                    \s*
                    $
                    """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineWorkTag]:
        tags = []
        for mrd_text_line in mrd_block:
            line_text = mrd_text_line.to_text("")
            match_obj = self.time_pattern.match(line_text)
            if match_obj:
                time_tag = (
                    mrd_text_line.id,
                    match_obj.start("time_range"),
                    match_obj.end("time_range"),
                    "timeRange",
                )
                tags.append(time_tag)
                title_tag = (
                    mrd_text_line.id,
                    match_obj.start("title"),
                    match_obj.end("title"),
                    "title",
                )
                tags.append(title_tag)
                depart_tag = (
                    mrd_text_line.id,
                    match_obj.start("depart"),
                    match_obj.end("depart"),
                    "depart",
                )
                tags.append(depart_tag)
            else:
                match_obj = self.com_pattern.match(line_text)
                if match_obj:
                    com_tag = (
                        mrd_text_line.id,
                        match_obj.start("company"),
                        match_obj.end("company"),
                        "company",
                    )
                    tags.append(com_tag)
        return tags


class QCWY2WorkBlockTagger:
    def __init__(self) -> None:
        re_year = r"\d{4}"
        re_month = "(1[012]|[1-9])"
        self.pattern1 = re.compile(
            rf"""
                ^
                (?P<time_range>{re_year}\s*/{re_month}\s*--{{1,2}}\s*({re_year}\s*/{re_month}|至今))
                :
                \s*
                (?P<company>.*)
                \s*
                (\(.*?\))
                \s*
                \[\s*\d+.*?\]
                $
                """,
            re.VERBOSE,
        )
        self.pattern11 = re.compile(
            rf"""
                ^
                (?P<time_range>{re_year}\s*/{re_month}\s*--{{1,2}}\s*({re_year}\s*/{re_month}|至今))
                :
                \s*
                (?P<company>.*)(\(\d+-\d+人\))?
                \s*
                \[\s*\d+.*?\]
                $
                """,
            re.VERBOSE,
        )
        self.pattern2 = re.compile(
            r"""
                    ^
                    职位名称:
                    \s*
                    (?P<title>.*?)
                    \s*
                    部(门|内):
                    (?P<depart>.*?)
                    \s*
                    $
                    """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineWorkTag]:
        tags = []
        for line_no, mrd_text_line in enumerate(mrd_block):
            line_text = mrd_text_line.to_text("")
            if ")[" in line_text or ") [" in line_text:
                if "外资(非欧美)" in line_text:
                    line_text = line_text.replace("外资(非欧美)", "外资$非欧美$")
                match_obj = self.pattern1.match(line_text)
            else:
                match_obj = self.pattern11.match(line_text)
            if match_obj:
                time_tag = (
                    mrd_text_line.id,
                    match_obj.start("time_range"),
                    match_obj.end("time_range"),
                    "timeRange",
                )
                tags.append(time_tag)
                com_tag = (
                    mrd_text_line.id,
                    match_obj.start("company"),
                    match_obj.end("company"),
                    "company",
                )
                tags.append(com_tag)
                next_line_text = mrd_block[line_no + 1].to_text("")
                if "所属行业" in next_line_text:
                    _line = mrd_block[line_no + 2]
                    if len(_line) == 2:
                        _span0_text = _line[0].text
                        _span1_text = _line[1].text
                        depart_tag = (_line.id, 0, len(_span0_text), "depart")
                        tags.append(depart_tag)
                        title_tag = (
                            _line.id,
                            len(_span0_text),
                            len(_span0_text) + len(_span1_text),
                            "title",
                        )
                        tags.append(title_tag)
            else:
                match_obj = self.pattern2.match(line_text)
                if match_obj:
                    title_tag = (
                        mrd_text_line.id,
                        match_obj.start("title"),
                        match_obj.end("title"),
                        "title",
                    )
                    tags.append(title_tag)
                    depart_tag = (
                        mrd_text_line.id,
                        match_obj.start("depart"),
                        match_obj.end("depart"),
                        "depart",
                    )
                    tags.append(depart_tag)
        return tags


class QCWY3WorkBlockTagger:
    def __init__(self) -> None:
        re_year = r"\d{4}"
        re_month = "(1[012]|[1-9])"
        self.pattern = re.compile(
            rf"""
                ^
                (?P<time_range>{re_year}\s*/{re_month}\s*-{{1,2}}\s*({re_year}\s*/{re_month}|至今))
                \s*
                (?P<company>.*\S)
                \s*
                \(.*?(月|年).*?\)
                \s*
                $
                """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineWorkTag]:
        tags = []
        for line_no, mrd_text_line in enumerate(mrd_block):
            line_text = mrd_text_line.to_text("")
            match_obj = self.pattern.match(line_text)
            if match_obj:
                time_tag = (
                    mrd_text_line.id,
                    match_obj.start("time_range"),
                    match_obj.end("time_range"),
                    "timeRange",
                )
                tags.append(time_tag)
                com_tag = (
                    mrd_text_line.id,
                    match_obj.start("company"),
                    match_obj.end("company"),
                    "company",
                )
                tags.append(com_tag)
                _line = mrd_block[line_no + 2]
                if len(_line) == 2:
                    _span0_text = _line[0].text
                    _span1_text = _line[1].text
                    depart_tag = (_line.id, 0, len(_span0_text), "depart")
                    tags.append(depart_tag)
                    title_tag = (
                        _line.id,
                        len(_span0_text),
                        len(_span0_text) + len(_span1_text),
                        "title",
                    )
                    tags.append(title_tag)
        return tags


class LiePin1WorkBlockTagger:
    def __init__(self) -> None:
        re_year = r"\d{4}"
        re_month = "(1[012]|0[1-9])"
        self.pattern1 = re.compile(
            rf"""
            ^
            (?P<time_range>{re_year}\.{re_month}\s*-\s*({re_year}\.{re_month}|至今))
            \s*
            (?P<comapny>.*?)
            $
            """,
            re.VERBOSE,
        )
        self.pattern2 = re.compile(
            rf"""
            ^
            (?P<time_range>{re_year}\.{re_month}\s*-\s?至?)
            \s*
            (?P<comapny>.*?)
            $
            """,
            re.VERBOSE,
        )
        self.pattern3 = re.compile(
            rf"""
            ^
            (?P<time_range>({re_year}\.{re_month}|至今|今))
            $
            """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineWorkTag]:
        tags = []
        for line_no, mrd_text_line in enumerate(mrd_block):
            line_text = mrd_text_line.to_text("")
            match_obj = self.pattern1.match(line_text)
            if match_obj:
                time_tag = (
                    mrd_text_line.id,
                    match_obj.start("time_range"),
                    match_obj.end("time_range"),
                    "timeRange",
                )
                tags.append(time_tag)
                com_tag = (
                    mrd_text_line.id,
                    match_obj.start("comapny"),
                    match_obj.end("comapny"),
                    "company",
                )
                tags.append(com_tag)
                if line_no + 1 < len(mrd_block):
                    next_line = mrd_block[line_no + 1]
                    next_line_text = next_line.to_text("")
                    title_tag = (next_line.id, 0, len(next_line_text), "title")
                    tags.append(title_tag)
            else:
                match_obj = self.pattern2.match(line_text)
                if match_obj:
                    if line_no + 1 >= len(mrd_block):
                        continue
                    next_line = mrd_block[line_no + 1]
                    next_line_text = next_line.to_text("")
                    next_match_obj = self.pattern3.match(next_line_text)
                    if not next_match_obj:
                        continue
                    time_tag = (
                        mrd_text_line.id,
                        match_obj.start("time_range"),
                        match_obj.end("time_range"),
                        "timeRange",
                    )
                    tags.append(time_tag)
                    com_tag = (
                        mrd_text_line.id,
                        match_obj.start("comapny"),
                        match_obj.end("comapny"),
                        "company",
                    )
                    tags.append(com_tag)
                    next_time_tag = (
                        next_line.id,
                        0,
                        len(next_line_text),
                        "M-timeRange",
                    )
                    tags.append(next_time_tag)
                    next_next_line = mrd_block[line_no + 2]
                    title_tag = (
                        next_next_line.id,
                        0,
                        len(next_next_line.to_text("")),
                        "title",
                    )
                    tags.append(title_tag)
        return tags


class LiePin2WorkBlockTagger:
    def __init__(self) -> None:
        re_year = r"\d{4}"
        re_month = "(1[012]|0[1-9])"
        self.pattern1 = re.compile(
            rf"""
            ^
            (?P<time_range>{re_year}/{re_month}\s*-\s*({re_year}/{re_month}|至今))
            \s*
            (?P<comapny>.*?)
            $
            """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineWorkTag]:
        tags = []
        for line_no, mrd_text_line in enumerate(mrd_block):
            line_text = mrd_text_line.to_text("")
            match_obj = self.pattern1.match(line_text)
            if match_obj:
                time_tag = (
                    mrd_text_line.id,
                    match_obj.start("time_range"),
                    match_obj.end("time_range"),
                    "timeRange",
                )
                tags.append(time_tag)
                com_tag = (
                    mrd_text_line.id,
                    match_obj.start("comapny"),
                    match_obj.end("comapny"),
                    "company",
                )
                tags.append(com_tag)
                for line in mrd_block[line_no + 1 :]:
                    if "bold" in line[0].font.lower():
                        title_tag = (line.id, 0, len(line.to_text("")), "title")
                        tags.append(title_tag)
                        break
        return tags


class Lagou1WorkBlockTagger:
    def __init__(self) -> None:
        re_year = r"\d{4}"
        re_month = "(1[012]|0[1-9])"
        self.pattern = re.compile(
            rf"""
            ^
            (?P<company>..*?)
            (?P<time_range>{re_year}\.{re_month}\s*-\s*({re_year}\.{re_month}|至今))
            $
            """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineWorkTag]:
        tags = []
        for line_no, mrd_text_line in enumerate(mrd_block):
            line_text = mrd_text_line.to_text("")
            match_obj = self.pattern.match(line_text)
            if match_obj:
                time_tag = (
                    mrd_text_line.id,
                    match_obj.start("time_range"),
                    match_obj.end("time_range"),
                    "timeRange",
                )
                tags.append(time_tag)
                com_tag = (
                    mrd_text_line.id,
                    match_obj.start("company"),
                    match_obj.end("company"),
                    "company",
                )
                tags.append(com_tag)
                if line_no + 1 < len(mrd_block):
                    next_line = mrd_block[line_no + 1]
                    next_line_text = next_line.to_text("")
                    if next_line_text.startswith("职位"):
                        title_tag = (next_line.id, 3, len(next_line_text), "title")
                        tags.append(title_tag)
        return tags


class Lagou2WorkBlockTagger:
    def __init__(self) -> None:
        re_year = r"\d{4}"
        re_month = "(1[012]|0[1-9])"
        self.pattern = re.compile(
            rf"""
            ^
            (?P<company>..*?)
            \s*
            (/(?P<depart>.*?))?
            \s*
            (?P<time_range>{re_year}\.{re_month}\s*-\s*({re_year}\.{re_month}|至今))
            $
            """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineWorkTag]:
        tags = []
        for line_no, mrd_text_line in enumerate(mrd_block):
            line_text = mrd_text_line.to_text("")
            match_obj = self.pattern.match(line_text)
            if match_obj:
                time_tag = (
                    mrd_text_line.id,
                    match_obj.start("time_range"),
                    match_obj.end("time_range"),
                    "timeRange",
                )
                tags.append(time_tag)
                com_tag = (
                    mrd_text_line.id,
                    match_obj.start("company"),
                    match_obj.end("company"),
                    "company",
                )
                tags.append(com_tag)
                depart_tag = (
                    mrd_text_line.id,
                    match_obj.start("depart"),
                    match_obj.end("depart"),
                    "depart",
                )
                tags.append(depart_tag)
                if line_no + 1 < len(mrd_block):
                    next_line = mrd_block[line_no + 1]
                    span = next_line[0]
                    if "bold" in span.font.lower() and line_no + 2 < len(mrd_block):
                        next_next_line = mrd_block[line_no + 2]
                        title_tag = (
                            next_next_line.id,
                            0,
                            len(next_next_line.to_text("")),
                            "title",
                        )
                        tags.append(title_tag)
                    else:
                        title_tag = (
                            next_line.id,
                            0,
                            len(next_line.to_text("")),
                            "title",
                        )
                        tags.append(title_tag)
        return tags


class WorkTaggerDispatcher:
    def __init__(self) -> None:
        self.tagger_map: Dict[str, Tagger] = {
            "zhilian1": ZhiLian1WorkBlockTagger(),
            "zhilian2": ZhiLian2WorkBlockTagger(),
            "qcwy1": QCWY1WorkBlockTagger(),
            "qcwy2": QCWY2WorkBlockTagger(),
            "qcwy3": QCWY3WorkBlockTagger(),
            "liepin1": LiePin1WorkBlockTagger(),
            "liepin2": LiePin2WorkBlockTagger(),
            "lagou1": Lagou1WorkBlockTagger(),
            "lagou2": Lagou2WorkBlockTagger(),
        }
        re_year = r"\d{4}"
        re_month = "(1[012]|[1-9])"
        self.qcwy2_pattern = re.compile(
            rf"""
                ^
                (?P<time_range>{re_year}\s*/{re_month}\s*--{{1,2}}\s*({re_year}\s*/{re_month}|至今))
                :
                \s*
                (?P<company>.*?)
                \s*
                \[\s*\d+.*?\]
                $
                """,
            re.VERBOSE,
        )
        re_year = r"\d{4}"
        re_month = "(1[012]|[1-9])"
        self.qcwy3_pattern = re.compile(
            rf"""
                ^
                (?P<time_range>{re_year}\s*/{re_month}\s*-{{1,2}}\s*({re_year}\s*/{re_month}|至今))
                \s*
                (?P<company>.*?)
                \s*
                \(.*?月.*?\)
                \s*
                $
                """,
            re.VERBOSE,
        )
        re_year = r"\d{4}"
        re_month = "(1[012]|0[1-9])"
        self.lagou_pattern = re.compile(
            rf"""
                    ^
                    (?P<company>..*?)
                    \s*
                    (?P<time_range>{re_year}\.{re_month}\s*-\s*({re_year}\.{re_month}|至今))
                    $
                    """,
            re.VERBOSE,
        )

    def dispatch(self, mrd_block: MrdBlock, block_source_name: str) -> Tagger:
        if block_source_name.startswith("ZL"):
            if block_source_name == "ZL_C":
                return self.tagger_map["zhilian2"]
            else:
                return self.tagger_map["zhilian1"]
        elif block_source_name.startswith("QCWY"):
            for line in mrd_block:
                line_text = line.to_text("")
                if self.qcwy2_pattern.match(line_text):
                    return self.tagger_map["qcwy2"]
                elif self.qcwy3_pattern.match(line_text):
                    return self.tagger_map["qcwy3"]
            else:
                return self.tagger_map["qcwy1"]
        elif block_source_name.startswith("LP"):
            if block_source_name == "LP_A":
                return self.tagger_map["liepin1"]
            elif block_source_name == "LP_C":
                return self.tagger_map["liepin2"]
            else:
                raise ValueError("can not dispatch block")
        elif block_source_name.startswith("LG"):
            for line_no, line in enumerate(mrd_block):
                line_text = line.to_text("")
                if self.lagou_pattern.match(line_text):
                    next_line_text = mrd_block[line_no + 1].to_text("")
                    if next_line_text.startswith("职位"):
                        return self.tagger_map["lagou1"]
                    else:
                        return self.tagger_map["lagou2"]
            else:
                raise ValueError("can not dispatch block")
        else:
            raise ValueError("can not dispatch block")
