# -*- coding: utf-8 -*-
from dataclasses import dataclass
from io import BytesIO
from typing import List

from .header.core import Header
from .header.infer import infer_headers
from .mrd import MrdDoc
from .mrd.mrd_pb2 import Document
from .proj.tag_maker import LineProjTag, ProjTaggerDispatcher
from .source.infer import infer_source
from .work.tag_maker import LineWorkTag, MrdBlock, WorkTaggerDispatcher


@dataclass
class DocInfo:

    source: str
    headers: List[Header]
    work_tags: List[LineWorkTag]
    proj_tags: List[LineProjTag]


work_dispatcher = WorkTaggerDispatcher()
proj_dispatcher = ProjTaggerDispatcher()


def get_mrd_block_by_name(doc: MrdDoc, headers: List[Header], name: str) -> MrdBlock:
    block = []
    for idx, header in enumerate(headers):
        if name in header.cat:
            if idx + 1 < len(headers):
                block.extend(doc[header.start : headers[idx + 1].start])
            else:
                block.extend(doc[header.start :])
    return block


def infer_doc_info(doc: MrdDoc) -> DocInfo:
    source = infer_source(doc)
    headers = infer_headers(doc, source)
    work_block = get_mrd_block_by_name(doc, headers, "Work")
    if work_block:
        tagger = work_dispatcher.dispatch(work_block, source.name)
        work_tags = tagger.tag(work_block)
    else:
        work_tags = []

    proj_block = get_mrd_block_by_name(doc, headers, "Proj")
    if proj_block:
        tagger = proj_dispatcher.dispatch(proj_block, source.name)
        proj_tags = tagger.tag(proj_block)
    else:
        proj_tags = []

    return DocInfo(
        source=source.name, headers=headers, work_tags=work_tags, proj_tags=proj_tags
    )


def infer_mrd_block_from_mrd(mrd):
    # For medivh
    bio = BytesIO()
    pb_doc = Document()
    mrd.write(bio)
    bio.flush()
    bio.seek(0)
    pb_doc.ParseFromString(bio.read())
    doc = MrdDoc.from_pb_doc(pb_doc)
    if len(doc.lines) <= 10:
        return None, "OTHER", "zh-cn"

    source = infer_source(doc)
    if source.name == "OTHER":
        return None, "OTHER", "zh-cn"

    headers = infer_headers(doc, source)
    start_dummy_head = Header(0, "Dummy", "Basic")
    end_dummy_header = Header(len(doc.lines), "Dummy", "")
    mrd_dict = doc.to_dict()
    for i, j in zip([start_dummy_head] + headers, headers + [end_dummy_header]):
        start = i.start
        end = j.start
        for line in mrd_dict["objs"][start:end]:
            for span in line:
                span["cats"] = i.cat

    return mrd_dict, source.name, "zh-cn"
