from typing import Dict, List

from ..mrd import MrdDoc
from ..source.base_source import ALL_SOURCE, Source
from .core import Header, HeaderInfer
from .header_set import (
    LAGOU_HEADER_SET,
    LIEPIN_HEADER_SET,
    QCWY_HEADER_SET,
    ZHILIAN_HEADER_SET,
)

_zl_header_infer = HeaderInfer(ZHILIAN_HEADER_SET)
_lg_header_infer = HeaderInfer(LAGOU_HEADER_SET)
_lp_header_infer = HeaderInfer(LIEPIN_HEADER_SET)
_qcwy_header_infer = HeaderInfer(QCWY_HEADER_SET)


header_infer_map: Dict[Source, HeaderInfer] = {}
for source in ALL_SOURCE:
    if "ZL" in source.name:
        header_infer_map[source] = _zl_header_infer
    elif "LG" in source.name:
        header_infer_map[source] = _lg_header_infer
    elif "LP" in source.name:
        header_infer_map[source] = _lp_header_infer
    elif "QCWY" in source.name:
        header_infer_map[source] = _qcwy_header_infer


def infer_headers(doc: MrdDoc, source: Source) -> List[Header]:
    header_infer = header_infer_map[source]
    return header_infer.infer(doc)
