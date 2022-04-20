# -*- coding: utf-8 -*-
import json
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Sequence

from ..mrd import MrdDoc

Header = namedtuple("Header", "start name cat")
Headers = Sequence[Header]


class HeaderInfer:
    def __init__(self, header_set: set, fix_header: bool = True):
        self.header_set = header_set
        self.fix_header = fix_header
        self._name_to_cat_map = self._load_laskw_json()

    @staticmethod
    def _load_laskw_json() -> dict:
        json_path = Path(__file__).parent / "laskw.json"
        return json.loads(json_path.read_text())

    def name_to_cat(self, name: str) -> str:
        name = name.replace(":", "").replace(".", "")
        return self._name_to_cat_map.get(name, "")

    def infer(self, doc: MrdDoc) -> List[Header]:
        header_spans = []
        for line in doc:
            for span in line:
                if span.text in self.header_set:
                    header_spans.append(span)

        header_attr_count = defaultdict(int)
        for span in header_spans:
            header_attr = self.span_to_attr(span)
            header_attr_count[header_attr] += 1

        if len(header_attr_count) > 2:
            raise ValueError("num of header attr > 2")
        header_attr = sorted(
            list(header_attr_count.items()), key=lambda x: x[1], reverse=True
        )[0][0]

        if header_attr_count[header_attr] <= 2:
            raise ValueError("num of header_attr <= 2")

        headers: List[Header] = []
        for line in doc:
            for span in line:
                span_attr = self.span_to_attr(span)
                if span_attr != header_attr:
                    continue

                if self.fix_header and span.text not in self.header_set:
                    continue

                headers.append(Header(line.id, span.text, self.name_to_cat(span.text)))

        return headers

    @staticmethod
    def span_to_attr(span):
        return span.position.left, span.fsize, span.font, span.color
