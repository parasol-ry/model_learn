from dataclasses import dataclass
from typing import Iterable, List, Optional
import torch

from torch.utils.data import Dataset
import numpy as np
from transformers import PreTrainedTokenizer


@dataclass
class SpanRecord:
    tokens: List[str]
    token_ids: List[int]
    tags: Optional[List[str]] = None


@dataclass
class CommRecord:
    spans: List[SpanRecord]
    comm_matrix: np.ndarray


def pad_sequence(sequence: List[List[int]], pad_token_id: int) -> torch.Tensor:
    pad_input_ids = []
    max_length = max(len(i) for i in sequence)
    for i in sequence:
        pad_input_ids.append(i + [pad_token_id] * (max_length - len(i)))
    return torch.tensor(pad_input_ids, dtype=torch.long)


def truncate_sequence(sequence: List[List[int]], max_length: int) -> List[List[int]]:
    return [i[: (max_length - 2)] for i in sequence]


def add_special_token(sequence: List[List[int]], start_token: int, end_token: int):
    new_token_ids = []
    for i in sequence:
        new_token_ids.append([start_token] + i + [end_token])
    return new_token_ids


class CommCollator:
    def __init__(self, tags: List[str], max_length: int = 64):
        self.max_length = max_length
        self.tags = tags
        self._tag2int = {tag: idx for idx, tag in enumerate(tags)}

    def tag2int(self, tag: str) -> int:
        return self._tag2int[tag]

    def int2tag(self, _int: int) -> str:
        return self.tags[_int]

    def __call__(self, records: List[CommRecord]):
        if len(records) != 1:
            raise ValueError("")

        record = records[0]
        token_ids = [i.token_ids for i in record.spans]
        token_ids = truncate_sequence(token_ids, self.max_length)
        token_ids = add_special_token(token_ids, 4, 3)  # CLS_ID: 4, SEP_ID: 3
        token_ids = pad_sequence(token_ids, 0)
        attention_mask = torch.where(
            token_ids != 0, torch.ones_like(token_ids), torch.zeros_like(token_ids)
        )
        comm_matrix = torch.from_numpy(record.comm_matrix)
        comm_matrix[comm_matrix < 0.1] = 0
        comm_matrix = comm_matrix / comm_matrix.sum(dim=-1)
        assert comm_matrix.size(0) == token_ids.size(0)
        output = {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "comm_matrix": comm_matrix,
        }

        have_tag = record.spans[0].tags is not None
        if have_tag:
            token_tags = [[self.tag2int(j) for j in i.tags] for i in record.spans]
            token_tags = truncate_sequence(token_tags, self.max_length)
            token_tags = add_special_token(token_tags, -100, -100)
            token_tags = pad_sequence(token_tags, -100)
            assert token_ids.size() == token_tags.size()
            output["tags"] = token_tags
        return output
