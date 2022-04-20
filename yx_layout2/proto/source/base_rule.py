from typing import ClassVar, Optional

from ..mrd import MrdDoc, MrdTextLikeObj
from .base_source import Source


class Rule:
    dep: ClassVar[Optional[Source]] = None

    def apply(self, **kwargs) -> Optional[Source]:
        raise NotImplemented


class DocRule(Rule):
    type_name: ClassVar[str] = "doc"

    def apply(self, doc: MrdDoc) -> Optional[Source]:
        raise NotImplemented


class SpanRule(Rule):
    type_name: ClassVar[str] = "span"

    def apply(self, span: MrdTextLikeObj) -> Optional[Source]:
        raise NotImplemented


class DocTextRule(Rule):
    type_name: ClassVar[str] = "doc_text"

    def apply(self, doc_text: str) -> Optional[Source]:
        raise NotImplemented


class LineTextRule(Rule):
    type_name: ClassVar[str] = "line_text"

    def apply(self, line_text: str) -> Optional[Source]:
        raise NotImplemented
