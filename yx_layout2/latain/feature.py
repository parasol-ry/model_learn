import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

# from moka_tokenizer import moka_codec

if TYPE_CHECKING:
    from .core import Block, Line, Span


# Feature: Bool, float, int, Category (str)
# Span 与 Span 之间的任务需要， Span 整体 Feature 之间的交互，这种交互依赖于 Feature 的类型
# Bool 和 Category 会判断是否相同
# float int 会转成 差异
# 例如：
# 这一行的 span 数量与对比行的 span 数量之间的 差异
# 这一行的上一行 span 数量与对比行的上一行 span 数量之间的 差异


# ===========  Line Feature  ===========
# Line Feature
# Line Feature，Prev Line Feature, Next Line Feature
# Line Feature 会有默认值，when line is None


def line_max_interspace(line: "Line") -> float:
    if len(line) <= 1:
        return 0
    return max(span.features["span_left_margin"] for span in line.data[1:])


def line_average_interspace(line: "Line") -> float:
    if len(line) <= 1:
        return 0
    return sum(span.features["span_left_margin"] for span in line.data[1:]) / (
        len(line) - 1
    )


# Pos Feature
def get_line_pos_feature(block: "Block", line_id: int) -> dict:
    line: "Line" = block[line_id]

    return {
        "line_left_margin": line.left - block.left,
        "line_right_margin": block.right - line.right,
        "line_top_margin": (line.top - line.prev_line.down)
        if line.prev_line is not None
        else 0,
        "line_down_margin": (line.next_line.top - line.down)
        if line.next_line is not None
        else 0,
        "line_is_most_left": block.left == line.left,
        "line_is_most_right": block.right == line.right,
        "line_is_align_with_prev": (line.left == line.prev_line.left)
        if line.prev_line
        else False,
        "line_is_align_with_next": (line.left == line.next_line.left)
        if line.next_line
        else False,
        "line_max_width": max(span.features["span_width"] for span in line),
        "line_max_interspace": line_max_interspace(line),
        "line_average_width": sum(span.features["span_width"] for span in line)
        / len(line),
        "line_average_interspace": line_average_interspace(line),
    }


def line_is_same_fsize(line: "Line") -> bool:
    fsize = line[0].fsize
    for span in line:
        if span.fsize != fsize:
            return False
    return True


def line_is_same_font(line: "Line") -> bool:
    font = line[0].font
    for span in line:
        if span.font != font:
            return False
    return True


def line_is_same_color(line: "Line") -> bool:
    color = line[0].color
    for span in line:
        if span.color != color:
            return False
    return True


def get_line_basic_feature(block: "Block", line_id: int) -> Dict[str, Any]:
    """TODO"""
    line: "Line" = block[line_id]
    return {
        # "line_is_first": line_id == 0,
        # "line_is_last": line_id == len(block) - 1,
        "line_id": str(line_id),
        "line_leading_fsize": line.fsize,
        "line_max_fsize": max(span.fsize for span in line),
        "line_is_same_fsize": line_is_same_fsize(line),
        "line_is_same_fsize_with_prev": (line.fsize == line.prev_line.fsize)
        if line.prev_line is not None
        else True,
        "line_is_same_fsize_with_next": (line.fsize == line.next_line.fsize)
        if line.next_line is not None
        else True,
        "line_is_same_font": line_is_same_font(line),
        "line_is_same_color": line_is_same_color(line),
        "line_num_spans": len(line),
    }


def line_have_time_range(line: "Line") -> bool:
    for span in line:
        if span.features["span_text_is_time_range"]:
            return True
    return False


def get_line_text_feature(block: "Block", line_id: int) -> dict:
    line: "Line" = block[line_id]
    line_num_tokens = sum(span.features["span_num_tokens"] for span in line)
    line_is_startswith_chinese = line[0].features["span_is_startswith_chinese"]
    return {
        "line_num_tokens": line_num_tokens,
        "line_is_startswith_chinese": line_is_startswith_chinese,
        "line_have_time_range": line_have_time_range(line),
    }


def get_line_feature(block: "Block", line_id: int) -> dict:
    return {
        **get_line_basic_feature(block, line_id),
        **get_line_pos_feature(block, line_id),
        **get_line_text_feature(block, line_id),
    }


# ===========  Span Feature  ===========
# Pos Feature
# Span Left , Span right
def span_left_margin(block: "Block", span: "Span") -> float:
    if span.prev_span is None:
        prev_span_right = block.left
    else:
        prev_span_right = span.prev_span.right
    return span.left - prev_span_right


def span_right_margin(block: "Block", span: "Span") -> float:
    if span.next_span is None:
        next_span_left = block.right
    else:
        next_span_left = span.next_span.left
    return next_span_left - span.right


def span_is_align_with_prev_line(block: "Block", line_id: int, span_id: int) -> bool:
    # 上一行同一位置的 Span 是否对齐，对齐按照 “left 是否相等” 判断
    # left 和 相等 这两个条件都应该进一步思考，相等是不是太严格了，left 对吗？ right 呢？
    line: "Line" = block[line_id]
    if line.prev_line is None:
        return False
    if len(line) != len(line.prev_line):
        return False
    if line[span_id].left == line.prev_line[span_id].left:
        return True
    return False


def span_is_align_with_next_line(block: "Block", line_id: int, span_id: int) -> bool:
    line: "Line" = block[line_id]
    if line.next_line is None:
        return False
    if len(line) != len(line.next_line):
        return False
    if line[span_id].left == line.next_line[span_id].left:
        return True
    return False


def span_text_leading_tag(token_tags) -> str:
    if len(token_tags) == 0:
        return "<None>"
    counter = Counter(token_tags)
    leading_tag = counter.most_common(1)[0][0]
    return leading_tag


# _re_year = r"\d{4}"
# _re_month = r"(1[012]|0[1-9])"
# _time_range_pattern = re.compile(
#     fr"^{_re_year}(\.|/){_re_month}-+({_re_year}(\.|/){_re_month}|至今)$"
# )


def span_text_is_time_range(token_tags: Sequence[str]) -> bool:
    num_tmr = 0
    for i in token_tags:
        if "TMR" in i:
            num_tmr += 1
    if num_tmr / len(token_tags) > 0.4:
        return True
    else:
        return False
    # return bool(_time_range_pattern.fullmatch(text.replace(" ", "")))


def span_normalized_fsize(block: "Block", span: "Span") -> float:
    if block.max_fsize == block.min_fsize:
        return 1

    return (span.fsize - block.min_fsize) / (block.max_fsize - block.min_fsize)


def get_span_basic_feature(block: "Block", line_id: int, span_id: int) -> dict:
    span = block[line_id][span_id]
    return {
        "span_font": span.font,
        "span_is_bold": "bold" in span.font.lower(),
        "span_color": ",".join(str(i) for i in span.color),
        "span_normalized_fsize": span_normalized_fsize(block, span),
        "span_id": str(span_id),
    }


def get_span_pos_feature(block: "Block", line_id: int, span_id: int) -> dict:
    line: "Line" = block[line_id]
    span: "Span" = line[span_id]

    return {
        "span_width": span.right - span.left,
        "span_left": span.left,
        "span_right": span.right,
        "span_is_most_left": span.left == block.left,
        "span_is_most_right": span.right == block.right,
        "span_left_margin": span_left_margin(block, span),
        "span_right_margin": span_right_margin(block, span),
        "span_is_align_with_prev_line": span_is_align_with_prev_line(
            block, line_id, span_id
        ),
        "span_is_align_with_next_line": span_is_align_with_next_line(
            block, line_id, span_id
        ),
    }


def get_span_text_feature(block: "Block", line_id: int, span_id: int) -> dict:
    span: "Span" = block[line_id][span_id]
    tokens = span.tokens
    token_tags = span.token_tags

    if tokens:
        first_token = tokens[0]
        last_token = tokens[-1]
    else:
        first_token = None
        last_token = None

    return {
        "span_num_tokens": len(tokens),
        "span_is_startswith_chinese": "CJK" in token_tags[0] if first_token else False,
        "span_first_token": first_token if first_token else "N",
        "span_first_token_tag": token_tags[0] if first_token else "N",
        "span_last_token": last_token if last_token else "N",
        "span_last_token_tag": token_tags[-1] if last_token else "N",
        "span_leading_tag": span_text_leading_tag(token_tags),
        "span_text_is_time_range": span_text_is_time_range(token_tags),
    }


def get_span_feature(block: "Block", line_id: int, span_id: int) -> dict:
    return {
        **get_span_basic_feature(block, line_id, span_id),
        **get_span_pos_feature(block, line_id, span_id),
        **get_span_text_feature(block, line_id, span_id),
    }


def str_compare_feature(
    value: Optional[str], other_value: Optional[str]
) -> Optional[float]:
    if value is not None and other_value is not None:
        # value_char_set = set(value)
        # other_value_char_set = set(other_value)
        # return len(value_char_set & other_value_char_set) / min(len(value_char_set), len(other_value_char_set))
        if value == other_value:
            return True
        else:
            return False
    return None


def bool_compare_feature(
    value: Optional[bool], other_value: Optional[bool]
) -> Optional[bool]:
    if value is not None and other_value is not None:
        if value == other_value:
            return True
        else:
            return False
    return None


def float_compare_feature(
    value: Optional[float], other_value: Optional[float]
) -> Optional[float]:
    if value is not None and other_value is not None:
        return abs(value - other_value)
    return None


def get_compare_features(
    features: Dict[str, Any], other_features: Dict[str, Any]
) -> Dict[str, Any]:
    compare_features = {}
    for k in features:
        value1 = features[k]
        value2 = other_features[k]
        if isinstance(value1, bool) or isinstance(value2, bool):
            compare_features[k] = bool_compare_feature(value1, value2)
        elif isinstance(value1, (float, int)) or isinstance(value2, (float, int)):
            compare_features[k] = float_compare_feature(value1, value2)
        elif isinstance(value1, str) or isinstance(value2, str):
            compare_features[k] = str_compare_feature(value1, value2)
        elif value1 is None and value2 is None:
            compare_features[k] = None
        else:
            raise TypeError(f"Unsupported type {type(value1)} and {type(value2)}")
    return compare_features
