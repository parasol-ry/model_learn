from collections import Counter, UserList, namedtuple
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .feature import get_line_feature, get_span_feature

Pos = namedtuple("Position", ("left", "top", "right", "down"))
Color = namedtuple("Color", ("r", "g", "b"))


def remove_tag_prefix(tag: str) -> str:
    """移除 token tag 上的前缀

    >>> remove_tag_prefix("B-timeRange")
    'timeRange'
    >>> remove_tag_prefix("O")
    'O'
    """
    if tag == "O":
        return tag

    return tag[2:]


def token_tags_to_sequence_tag(token_tags: Iterable[str]) -> str:
    """将 Token Tag 序列转换成 Sequence Tag，忽略 O 标签

    >>> token_tags_to_sequence_tag(["B-timeRange", "I-timeRange", "I-timeRange", "O"])
    'timeRange'
    >>> token_tags_to_sequence_tag(["B-timeRange", "I-timeRange", "B-company", "I-company"])
    'company|timeRange'
    >>> token_tags_to_sequence_tag(["O", "O", "O])
    ''

    Args:
        token_tags (Iterable[str]): token 序列

    Returns:
        str: Sequence Tag
    """
    processed_tags = [remove_tag_prefix(i) for i in token_tags]
    processed_tags = sorted(list(set([i for i in processed_tags if i != "O"])))
    return "|".join(processed_tags)


class PosMixin:
    pos: Pos

    @property
    def left(self) -> float:
        return self.pos.left

    @property
    def right(self) -> float:
        return self.pos.right

    @property
    def top(self) -> float:
        return self.pos.top

    @property
    def down(self) -> float:
        return self.pos.down


@dataclass
class Span(PosMixin):
    pos: Pos
    fsize: float
    font: str
    color: Color
    tokens: List[str]
    token_tags: List[str]
    text: str
    tag: str
    prev_span: Optional["Span"] = field(default=None, repr=False)
    next_span: Optional["Span"] = field(default=None, repr=False)
    features: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, xuen_dict: dict) -> "Span":
        pos = Pos(*xuen_dict["pos"])
        color = Color(*xuen_dict["color"])
        text = "".join([i[0] for i in xuen_dict["text"]])
        tokens = [i[0] for i in xuen_dict["text"]]
        token_tags = [i[1] for i in xuen_dict["text"]]
        tag = token_tags_to_sequence_tag(xuen_dict["tags"])
        return Span(
            pos,
            xuen_dict["fsize"],
            xuen_dict["font"],
            color,
            tokens,
            token_tags,
            text,
            tag,
        )


class Line(UserList, PosMixin):
    data: List[Span]
    pos: Pos
    # 只实现了 append 不要使用除了 初始化和 append 外的任何会修改列表元素的方法

    def __init__(self, sorted_spans: Iterable[Span]):
        self.data = list(sorted_spans)
        if not self.data:
            raise ValueError("Line data cannot be empty")
        self.pos = Pos(
            self[0].pos.left,
            min([i.pos.top for i in sorted_spans]),
            self[-1].pos.right,
            max([i.pos.down for i in sorted_spans]),
        )
        self.link_spans()
        self.fsize = self.get_leading_fsize()
        self.prev_line: Optional["Line"] = None
        self.next_line: Optional["Line"] = None
        self.features: dict = {}

    def get_leading_fsize(self):
        counter = Counter(i.fsize for i in self)
        return counter.most_common(1)[0][0]

    def link_spans(self):
        if len(self) <= 1:
            return

        tmp_span = self[0]
        tmp_span.prev_span = None
        for i in self[1:]:
            i.prev_span = tmp_span
            tmp_span.next_span = i
            tmp_span = i

    def append(self, item: Span) -> None:
        if self:
            last_item = self[-1]
            last_item.next_span = item
            item.prev_span = last_item
        self.data.append(item)


class Block(UserList, PosMixin):
    data: List[Line]
    pos: Pos

    def __init__(self, lines: List[Line], file: Optional[str] = None):
        self.file = file
        self.data = lines
        self.link_lines()
        self.pos = Pos(
            min([j.pos.left for i in self for j in i]),
            min([j.pos.top for j in self[0]]),
            max([j.pos.right for i in self for j in i]),
            max([j.pos.down for j in self[-1]]),
        )
        self.max_fsize = max([j.fsize for i in self for j in i])
        self.min_fsize = min([j.fsize for i in self for j in i])
        self._featured = False

    @property
    def featured(self) -> bool:
        return self._featured

    def link_lines(self) -> None:
        if len(self) <= 1:
            return
        tmp_line = self[0]
        tmp_line.prev_line = None
        for i in self[1:]:
            i.prev_line = tmp_line
            tmp_line.next_line = i
            tmp_line = i

    @classmethod
    def from_dict(cls, xuen_dict: dict) -> "Block":
        """将 xuen_dict 转化为 Block"""
        lines = []
        for raw_line in xuen_dict["objs"]:
            spans = []
            for raw_span in raw_line:
                span = Span.from_dict(raw_span)
                spans.append(span)
            if spans:
                line = Line(spans)
                lines.append(line)
        return Block(
            file=xuen_dict["file"],
            lines=lines,
        )

    def generate_feature(self) -> None:
        """生成 Block 的特征"""
        for line_id, line in enumerate(self):
            for span_id, span in enumerate(line):
                span.features = get_span_feature(self, line_id, span_id)
        for line_id, line in enumerate(self):
            line.features = get_line_feature(self, line_id)
        self._featured = True
