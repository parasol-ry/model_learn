import json
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, SupportsIndex, Union, overload

from moka_tokenizer import normalize

from .mrd_pb2 import Document

Position = namedtuple("Position", ("left", "top", "right", "down"))
Color = namedtuple("Color", ("r", "g", "b"))


def int_to_color(value: int) -> Color:
    # rgb: (102, 119, 135) -> 6715271
    # 102 --[bin]--> 0b01100110, 119 --[bin]--> 0b01110111, 135 --[bin]--> 0b10000111
    # 0b011001100111011110000111 --[int]--> 6715271
    # 0xff, last Byte
    return Color((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


@dataclass
class MrdObj:
    position: Position = Position(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_pb_obj(cls, pb_obj):
        raise NotImplementedError

    @staticmethod
    def get_position_from_pb_obj(pb_obj) -> Position:
        return Position(
            pb_obj.rect.x,
            pb_obj.rect.y,
            pb_obj.rect.x + pb_obj.rect.width,
            pb_obj.rect.y + pb_obj.rect.height,
        )

    def iomin(self, other_obj: "MrdObj") -> float:
        # intersection over min_height
        intersection = max(
            0,
            min(self.position[3], other_obj.position[3])
            - max(self.position[1], other_obj.position[1]),
        )
        min_ = min(
            max(2, self.position[3] - self.position[1]),
            max(2, other_obj.position[3] - other_obj.position[1]),
        )
        return intersection / min_

    def hr(self, other_obj: "MrdObj") -> float:
        # height ratio
        h1 = max(2, self.position[3] - self.position[1])
        h2 = max(2, other_obj.position[3] - other_obj.position[1])
        return min(h1, h2) / max(h1, h2)

    def is_same_line(
        self,
        other_obj: "MrdObj",
        iomin_threshold: float = 0.65,
        hr_threshold: float = 0.7,
    ) -> bool:
        if other_obj.position.top > self.position.down:
            return False
        iomin = self.iomin(other_obj)
        hr = self.hr(other_obj)
        if iomin > iomin_threshold and hr > hr_threshold:
            return True
        return False


@dataclass
class MrdText(MrdObj):
    font: str = ""
    fsize: int = 0
    color: Color = Color(0, 0, 0)
    text: str = ""
    id: int = -1000

    @classmethod
    def from_pb_obj(cls, pb_obj):
        return cls(
            position=cls.get_position_from_pb_obj(pb_obj),
            font=pb_obj.font,
            fsize=pb_obj.fontSize,
            color=int_to_color(pb_obj.fontColor),
            text=normalize(pb_obj.unicode.strip()).strip(),
        )


@dataclass
class MrdOcr(MrdObj):
    font: str = ""
    fsize: int = 0
    color: Color = Color(0, 0, 0)
    text: str = ""
    id: int = -1000

    @classmethod
    def from_pb_obj(cls, pb_obj):
        return cls(
            position=cls.get_position_from_pb_obj(pb_obj),
            font=pb_obj.font,
            fsize=pb_obj.fontSize,
            color=int_to_color(pb_obj.fontColor),
            text=normalize(pb_obj.unicode.strip()).strip(),
        )


@dataclass
class MrdImage(MrdObj):
    type_: int = 0
    category: int = 0
    data: bytes = b""

    @classmethod
    def from_pb_obj(cls, pb_obj):
        return cls(
            position=cls.get_position_from_pb_obj(pb_obj),
            type_=pb_obj.type,
            category=pb_obj.category,
            data=pb_obj.data,
        )


@dataclass
class MrdStruct(MrdObj):
    structs: list = field(default_factory=list)

    @classmethod
    def from_pb_obj(cls, pb_obj):
        OBJ_MAPPING = {
            "textObject": MrdText,
            "ocrResultObject": MrdOcr,
            "imageObject": MrdImage,
            "structObject": MrdStruct,
        }
        structs = []
        for pb_obj in pb_obj.structs:
            pb_obj_type = pb_obj.WhichOneof("object")
            structs.append(
                OBJ_MAPPING[
                    pb_obj[pb_obj_type].from_pb_obj(getattr(pb_obj, pb_obj_type))
                ]
            )
        return cls(
            position=cls.get_position_from_pb_obj(pb_obj),
            structs=structs,
        )


MrdTextLikeObj = Union[MrdText, MrdOcr]


@dataclass
class MrdTextLine:
    spans: List[MrdTextLikeObj] = field(default_factory=list)
    id: int = -1000  # offset， -1000 is None

    def __post_init__(self):
        self.position = Position(
            min(i.position.left for i in self.spans),
            min(i.position.top for i in self.spans),
            max(i.position.right for i in self.spans),
            max(i.position.down for i in self.spans),
        )

    def __getitem__(self, item) -> MrdTextLikeObj:
        return self.spans[item]

    def __len__(self):
        return len(self.spans)

    def to_text(self, sep=" ") -> str:
        return sep.join(i.text for i in self)


class MrdDoc:
    OBJ_MAPPING = {
        "textObject": MrdText,
        "ocrResultObject": MrdOcr,
        "imageObject": MrdImage,
        "structObject": MrdStruct,
    }

    def __init__(
        self,
        file: str,
        lines: Iterable[MrdTextLine],
    ) -> None:
        self.file = file
        self.lines = list(lines)
        self.reset_id()

    def reset_id(self):
        for line_offset, line in enumerate(self.lines):
            line.id = line_offset
            for span_offset, span in enumerate(line.spans):
                span.id = span_offset

    @overload
    def __getitem__(self, i: SupportsIndex) -> MrdTextLine:
        ...

    @overload
    def __getitem__(self, i: slice) -> List[MrdTextLine]:
        ...

    def __getitem__(
        self, i: Union[SupportsIndex, slice]
    ) -> Union[MrdTextLine, List[MrdTextLine]]:
        return self.lines[i]

    def __len__(self):
        return len(self.lines)

    @classmethod
    def from_pb_file(cls, file: Union[str, Path]) -> "MrdDoc":
        file = str(file)
        pb_doc = Document()
        with open(file, "rb") as f:
            pb_doc.ParseFromString(f.read())  # type: ignore
        return cls.from_pb_doc(pb_doc)

    @classmethod
    def from_pb_doc(cls, pb_doc: Document) -> "MrdDoc":
        page_height_offset: float = 0
        lines = []

        for pb_page in pb_doc.pages:  # type: ignore
            text_objs = []
            non_text_objs = []
            for pb_obj in pb_page.objects:
                pb_obj_type = pb_obj.WhichOneof("object")
                mrd_obj = cls.OBJ_MAPPING[pb_obj_type].from_pb_obj(
                    getattr(pb_obj, pb_obj_type)
                )
                if pb_obj_type == "textObject" or pb_obj_type == "ocrResultObject":
                    if mrd_obj.text:
                        text_objs.append(mrd_obj)
                else:
                    non_text_objs.append(mrd_obj)
            for text_obj in text_objs:
                # text_obj.position = Position(*(i + page_height_offset for i in text_obj.position))
                text_obj.position = Position(
                    text_obj.position[0],
                    text_obj.position[1] + page_height_offset,
                    text_obj.position[2],
                    text_obj.position[3] + page_height_offset,
                )
            if text_objs:
                line_texts = cls.group_mrd_text_objs(text_objs)
                head = "\n".join(line_obj.to_text("") for line_obj in line_texts[:15])
                tail = "\n".join(line_obj.to_text("") for line_obj in line_texts[-10:])
                # lang = detectMrdLangfromStr(head, tail)
                lang = "zh-cn"
                if lang == "zh-cn":
                    lines.extend(line_texts)
                else:
                    break

            page_height_offset = lines[-1].position.down
        return cls(
            file=pb_doc.name,  # type: ignore
            lines=lines,
        )

    @classmethod
    def from_dict(cls, record: dict) -> "MrdDoc":
        # untagged record
        file = record.get("file", "")

        text_lines: List[MrdTextLine] = []
        for line in record["objs"]:
            mrd_text_objs = []
            for obj in line:
                obj = MrdText(
                    position=Position(*obj["pos"]),
                    font=obj["font"],
                    color=obj["color"],
                    fsize=obj["fsize"],
                    text=obj["text"],
                )
                mrd_text_objs.append(obj)
            text_lines.append(MrdTextLine(mrd_text_objs))
        return MrdDoc(
            file=file,
            lines=text_lines,
        )

    def to_dict(self) -> dict:
        objs = []
        for line in self.lines:
            span_objs = []
            for span_obj in line.spans:
                span_objs.append(
                    {
                        "pos": list(span_obj.position),
                        "font": span_obj.font,
                        "color": list(span_obj.color),
                        "fsize": span_obj.fsize,
                        "text": span_obj.text,
                    }
                )
            objs.append(span_objs)
        return {"file": self.file, "objs": objs}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_text(self, span_sep=" ") -> str:
        return "\n".join(self.to_line_texts(span_sep))

    def to_line_texts(self, span_sep=" ") -> List[str]:
        line_texts = []
        for line in self:
            line_texts.append(line.to_text(span_sep))
        return line_texts

    @staticmethod
    def group_mrd_text_objs(text_objs: Sequence[MrdTextLikeObj]) -> List[MrdTextLine]:
        if not text_objs:
            return []
        sorted_mrd_objs = sorted(
            text_objs, key=lambda x: (x.position[1], x.position[0])
        )
        mrd_lines: List[MrdTextLine] = []

        cur_obj = sorted_mrd_objs[0]
        cur_line_objs = [cur_obj]
        for mrd_obj in sorted_mrd_objs[1:]:
            if cur_obj.is_same_line(mrd_obj):
                cur_line_objs.append(mrd_obj)
                cur_obj = mrd_obj
            else:
                cur_line_objs.sort(key=lambda x: x.position.left)
                mrd_lines.append(MrdTextLine(spans=cur_line_objs))
                cur_obj = mrd_obj
                cur_line_objs = [cur_obj]
        if cur_line_objs:
            cur_line_objs.sort(key=lambda x: x.position.left)
            mrd_lines.append(MrdTextLine(spans=cur_line_objs))
        mrd_lines.sort(key=lambda x: x.position.top)
        return mrd_lines

    def _format(self, o):
        if type(o) == list:
            return f"[{','.join(map(str, o))}]"
        else:
            return str(o)

    def __repr__(self):
        head = f"file: {self.file}\n"
        return head + self.to_text()
