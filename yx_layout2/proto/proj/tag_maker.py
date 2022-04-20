# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Protocol, Sequence

from ..mrd.core import MrdTextLine
from .record import LineProjTag

MrdBlock = Sequence[MrdTextLine]


class ProjTagger(Protocol):
    def tag(self, block: MrdBlock) -> List[LineProjTag]:
        ...


class ZhiLian1ProjBlockTagger:
    def __init__(self) -> None:
        self.pattern = re.compile(
            r"""
            ^
            (?P<time_range>\d{4}\.\d{2}\s*-\s*(\d{4}\.\d{2}|至今))
            (?P<project>.*?)
            $
            """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineProjTag]:
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
                if not match_obj.group("project"):
                    prev_line = mrd_block[line_no - 1]
                    _len = len(prev_line.text(""))
                    project_tag = (prev_line.id, 0, _len, "projectName")
                else:
                    project_tag = (
                        mrd_text_line.id,
                        match_obj.start("project"),
                        match_obj.end("project"),
                        "projectName",
                    )
                tags.append(project_tag)
        return tags


class ZhiLian2ProjBlockTagger:
    def __init__(self) -> None:
        self.pattern = re.compile(
            r"""
            ^
            (?P<time_range>\d{4}\.\d{2}\s*-\s*(\d{4}\.\d{2}|至今))
            (?P<project>.*?)
            (\(\d.*(年|月)\))?
            $
            """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineProjTag]:
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
                project_tag = (
                    mrd_text_line.id,
                    match_obj.start("project"),
                    match_obj.end("project"),
                    "projectName",
                )
                tags.append(project_tag)
        return tags


class QCWYProjBlockTagger:
    def __init__(self) -> None:
        re_year = "\d{4}"
        re_month = "(1[012]|[1-9])"
        self.pattern = re.compile(
            rf"""
                ^
                (?P<time_range>{re_year}\s*/{re_month}\s*-{{1,2}}\s*({re_year}\s*/{re_month}|至今))
                :?
                (?P<project>.*?)
                $
                """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineProjTag]:
        tags = []
        for line_no, mrd_text_line in enumerate(mrd_block):
            line_text = mrd_text_line.to_text("")
            if line_text.startswith("所属公司:"):
                com_tag = (mrd_text_line.id, 5, len(line_text), "company")
                tags.append(com_tag)
            else:
                match_obj = self.pattern.match(line_text)
                if match_obj:
                    time_tag = (
                        mrd_text_line.id,
                        match_obj.start("time_range"),
                        match_obj.end("time_range"),
                        "timeRange",
                    )
                    tags.append(time_tag)
                    project_tag = (
                        mrd_text_line.id,
                        match_obj.start("project"),
                        match_obj.end("project"),
                        "projectName",
                    )
                    tags.append(project_tag)
        return tags


class LiepinProjBlockTagger:
    def __init__(self) -> None:
        re_year = "\d{4}"
        re_month = "(1[012]|0[1-9])"
        self.pattern = re.compile(
            rf"""
            ^
            (?P<time_range>{re_year}\.{re_month}\s*-\s*({re_year}\.{re_month}|至今))
            (?P<project>.*?)
            $
            """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineProjTag]:
        tags = []
        for line_no, mrd_text_line in enumerate(mrd_block):
            line_text = mrd_text_line.to_text("")
            if line_text.startswith("- 项目职务:"):
                title_tag = (mrd_text_line.id, 7, len(line_text), "title")
                tags.append(title_tag)
            elif line_text.startswith("- 所在公司:"):
                com_tag = (mrd_text_line.id, 7, len(line_text), "company")
                tags.append(com_tag)
            else:
                match_obj = self.pattern.match(line_text)
                if match_obj:
                    time_tag = (
                        mrd_text_line.id,
                        match_obj.start("time_range"),
                        match_obj.end("time_range"),
                        "timeRange",
                    )
                    tags.append(time_tag)
                    project_tag = (
                        mrd_text_line.id,
                        match_obj.start("project"),
                        match_obj.end("project"),
                        "projectName",
                    )
                    tags.append(project_tag)
        return tags


class LagouProjBlockTagger:
    def __init__(self) -> None:
        re_year = "\d{4}"
        re_month = "(1[012]|0[1-9])"
        self.pattern = re.compile(
            rf"""
            ^
            (?P<project>..*?)
            (?P<time_range>{re_year}\.{re_month}\s*-\s*({re_year}\.{re_month}|至今))
            $
            """,
            re.VERBOSE,
        )

    def tag(self, mrd_block: MrdBlock) -> List[LineProjTag]:
        tags = []
        for line_no, mrd_text_line in enumerate(mrd_block):
            line_text = mrd_text_line.to_text("")
            if line_text.startswith("所属公司:("):
                com_tag = (mrd_text_line.id, 6, len(line_text) - 1, "company")
                tags.append(com_tag)
            else:
                match_obj = self.pattern.match(line_text)
                if match_obj:
                    time_tag = (
                        mrd_text_line.id,
                        match_obj.start("time_range"),
                        match_obj.end("time_range"),
                        "timeRange",
                    )
                    tags.append(time_tag)
                    project_tag = (
                        mrd_text_line.id,
                        match_obj.start("project"),
                        match_obj.end("project"),
                        "projectName",
                    )
                    tags.append(project_tag)
        return tags


class ProjTaggerDispatcher:
    def __init__(self) -> None:
        self.taggers: Dict[str, ProjTagger] = {
            "zl1": ZhiLian1ProjBlockTagger(),
            "zl2": ZhiLian2ProjBlockTagger(),
            "qcwy": QCWYProjBlockTagger(),
            "liepin": LiepinProjBlockTagger(),
            "lagou": LagouProjBlockTagger(),
        }

    def dispatch(self, mrd_block: MrdBlock, block_source_name: str) -> ProjTagger:
        if block_source_name.startswith("ZL"):
            for line in mrd_block[:5]:
                line_text = line.to_text("")
                if "项目经验" in line_text:
                    return self.taggers["zl1"]
                elif "项目经历" in line_text:
                    return self.taggers["zl2"]
            else:
                raise ValueError("can not dispatch block")
        elif block_source_name.startswith("QCWY"):
            for line in mrd_block[:5]:
                line_text = line.to_text("")
                if "项目经验" in line_text:
                    return self.taggers["qcwy"]
            else:
                raise ValueError("can not dispatch block")
        elif block_source_name.startswith("LP"):
            for line in mrd_block[:5]:
                line_text = line.to_text("")
                if "项目经历" in line_text:
                    return self.taggers["liepin"]
            else:
                raise ValueError("can not dispatch block")
        elif block_source_name.startswith("LG"):
            for line in mrd_block[:5]:
                line_text = line.to_text("")
                if "项目经验" in line_text or "项目经历" in line_text:
                    return self.taggers["lagou"]
            else:
                raise ValueError("can not dispatch block")
        else:
            raise ValueError("can not dispatch block")
