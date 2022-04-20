from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from transformers.models.bert import modeling_bert


class CommLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.9):
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: Tensor, comm_matrix: Tensor) -> Tensor:
        # hidden_states: (num_span, hidden_size)
        # comm_matrix: (num_span, num_span)
        # hidden_value: (num_span, hidden_size)
        hidden_value = self.linear(hidden_states)
        comm_matrix = self.dropout(comm_matrix)
        # comm_value: (num_span, hidden_size)
        comm_value = torch.mm(comm_matrix, hidden_value)
        final_value = hidden_states + self.layerNorm(comm_value + hidden_value)
        return final_value


class CommEncoder(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.9,
        comm_weight: float = 1,
        num_comm_layer: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.comm = nn.ModuleList(
            [CommLayer(hidden_size) for _ in range(num_comm_layer)]
        )
        self.num_classes = num_classes
        self.classifier = nn.Linear(2 * hidden_size, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.comm_weight = comm_weight

    def forward(
        self, input_ids, attention_mask, comm_matrix, label: Optional[Tensor] = None
    ):
        # input_ids: (num_span, sequence_lenth)
        # attention_mask: (num_span, sequence_lenth)
        # comm_matrix: (num_span, num_span)
        encoder_ouput = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # span_encoding: (num_span, hidden_size)
        span_encoding = encoder_ouput.last_hidden_state[:, 0, :]
        # comm_encoding: (num_span, hidden_size)
        comm_encoding = self.comm(span_encoding, comm_matrix)
        comm_encoding = comm_encoding * self.comm_weight
        # token_encoding: (num_span, sequence_length, hidden_size)
        token_encoding = encoder_ouput.last_hidden_state
        # mix_token_encoding: (num_span, sequence_length, hidden_size)
        mix_token_encoding = torch.cat([comm_encoding.unsqueeze(1), token_encoding])
        # logits: (num_span, sequence_length, num_classes)
        logits = self.classifier(mix_token_encoding)
        output_dict = {"logits": logits}
        if label is not None:
            loss = self.criterion(logits, label)
            output_dict["loss"] = loss
        return output_dict
