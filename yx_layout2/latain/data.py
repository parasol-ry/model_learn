from dataclasses import dataclass
from tkinter import N
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from latain.core import Block
from latain.feature import get_compare_features

Feature = Union[str, int, float, bool]
NormalizedLayout = Tuple[int, int, int, int]


@dataclass
class BlockHandcraftRecord:
    span_features: List[Dict[str, Feature]]
    file: Optional[str] = None
    span_pair_labels: Optional[List[List[int]]] = None
    span_tags: Optional[List[str]] = None

    def __post_init__(self):
        if self.span_pair_labels:
            assert (
                len(self.span_pair_labels)
                == len(self.span_pair_labels[0])
                == len(self.span_features)
            )

    @classmethod
    def from_block(cls, block: Block) -> "BlockHandcraftRecord":
        if not block.featured:
            block.generate_feature()

        span_features = []
        for line in block:
            cur_line_features = line.features
            line_feature_keys = list(cur_line_features.keys())
            if line.prev_line is None:
                prev_line_features = cls._get_none_features(line_feature_keys, "prev")
            else:
                prev_line_features = {
                    "prev_" + k: v for k, v in line.prev_line.features.items()
                }

            if line.next_line is None:
                next_line_features = cls._get_none_features(line_feature_keys, "next")
            else:
                next_line_features = {
                    "next_" + k: v for k, v in line.next_line.features.items()
                }

            line_level_features = {
                **cur_line_features,
                **prev_line_features,
                **next_line_features,
            }

            for span in line:
                cur_span_features = span.features
                span_feature_keys = list(cur_span_features.keys())
                if span.prev_span is None:
                    prev_span_features = cls._get_none_features(
                        span_feature_keys, "prev"
                    )
                else:
                    prev_span_features = {
                        "prev_" + k: v for k, v in span.prev_span.features.items()
                    }
                if span.next_span is None:
                    next_span_features = cls._get_none_features(
                        span_feature_keys, "next"
                    )
                else:
                    next_span_features = {
                        "next_" + k: v for k, v in span.next_span.features.items()
                    }
                span_level_features = {
                    **cur_span_features,
                    **prev_span_features,
                    **next_span_features,
                }
                features = {**line_level_features, **span_level_features}
                span_features.append(features)
        span_tags = [span.tag for line in block for span in line]
        span_pair_labels = []
        for idx, span_tag in enumerate(span_tags):
            pair_labels = []
            for other_tag in span_tags:
                if span_tag == other_tag:
                    pair_labels.append(1)
                else:
                    pair_labels.append(0)
            # pair_labels[idx] = 0
            span_pair_labels.append(pair_labels)
        return cls(
            file=block.file,
            span_features=span_features,
            span_pair_labels=span_pair_labels,
            span_tags=span_tags,
        )

    @staticmethod
    def _get_none_features(feature_keys: List[str], prefix: str) -> Dict[str, None]:
        return {prefix + "_" + k: None for k in feature_keys}

    def to_flat_records(self, filter_: Optional[Callable] = None):
        """生成 pair 对比特征"""
        if self.span_pair_labels is None:
            raise ValueError("span pair labels is not set")

        records = []
        ids = []
        for idx, span_features in enumerate(self.span_features):
            for other_idx in range(idx + 1, len(self.span_features)):
                flag = True
                other_span_features = self.span_features[other_idx]
                if self.span_tags is not None and filter_ is not None:
                    span_tag = self.span_tags[idx]
                    other_span_tag = self.span_tags[other_idx]
                    flag = filter_(idx, other_idx, span_tag, other_span_tag)
                if flag:
                    label = self.span_pair_labels[idx][other_idx]
                    features = get_compare_features(span_features, other_span_features)
                    record = {**features, "label": label}
                    ids.append(
                        (
                            (span_features["line_id"], span_features["span_id"], idx),
                            (
                                other_span_features["line_id"],
                                other_span_features["span_id"],
                                other_idx,
                            ),
                        )
                    )
                    records.append(record)
        return records, ids


@dataclass
class BlockLayoutRecord:
    span_layouts: List[NormalizedLayout]
    span_labels: Optional[List[List[int]]] = None

    def __post_init__(self):
        if self.span_labels:
            assert (
                len(self.span_labels)
                == len(self.span_labels[0])
                == len(self.span_layouts)
            )
