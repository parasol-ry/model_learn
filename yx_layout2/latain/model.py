import math
from dataclasses import dataclass
from tkinter import N
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class LayoutModelConfig:
    hidden_size: int = 256
    max_position_embeddings: int = 512
    max_lr_position_embeddings: int = 512
    max_td_position_embeddings: int = 1024
    real_2d_num_buckets: int = 128
    max_rel_2d_pos: int = 256
    layer_norm_eps: float = 1e-6
    dropout: float = 0.1
    num_attention_heads: int = 4
    intermediate_size: int = 512
    num_hidden_layers: int = 4
    initializer_range: float = 0.02


class LayoutEmbeeding(nn.Module):
    def __init__(self, config: LayoutModelConfig):
        coordinate_size = config.hidden_size // 6
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.l_position_embeddings = nn.Embedding(
            config.max_lr_position_embeddings, coordinate_size
        )
        self.r_position_embeddings = nn.Embedding(
            config.max_lr_position_embeddings, coordinate_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_lr_position_embeddings, coordinate_size
        )
        self.t_position_embeddings = nn.Embedding(
            config.max_td_position_embeddings, coordinate_size
        )
        self.d_position_embeddings = nn.Embedding(
            config.max_td_position_embeddings, coordinate_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_td_position_embeddings, coordinate_size
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def calc_layout_embedding(self, bbox: torch.Tensor) -> torch.Tensor:  # type: ignore
        # bbox -> output
        # bbox: (batch_size, num_spans, 4)
        # output: (batch_size, num_spans, hidden_size)

        l = self.l_position_embeddings(bbox[:, :, 0])
        r = self.r_position_embeddings(bbox[:, :, 2])

        t = self.t_position_embeddings(bbox[:, :, 1])
        d = self.d_position_embeddings(bbox[:, :, 3])

        w = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        h = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        embeddings = torch.cat(
            [
                l,
                r,
                t,
                d,
                w,
                h,
            ],
            dim=-1,
        )
        return embeddings


class LayoutSelfAttention(nn.Module):
    def __init__(self, config: LayoutModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_spans, hidden_size)
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rel_2d_pos: torch.Tensor,
        output_attentions: bool = False,
    ):
        # hidden_states: (batch_size, num_spans, hidden_size)
        # rel_pos: (batch_size, num_head, num_spans, num_spans)
        # rel_2d_pos: (batch_size, num_head, num_spans, num_spans)

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # query_layer: (batch_size, num_head, num_spans, head_hidden_size)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        # attention_scores: (batch_size, num_head, num_spans, num_spans)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores += rel_2d_pos
        attention_scores = attention_scores.float().masked_fill_(
            attention_mask.to(torch.bool), float("-inf")
        )

        # attention_probs: (batch_size, num_head, num_spans, num_spans)
        attention_probs = torch.softmax(
            attention_scores, dim=-1, dtype=torch.float32
        ).type_as(value_layer)
        attention_probs = self.dropout(attention_probs)

        # context_layer: (batch_size,, num_head, num_spans, head_hidden_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer: (batch_size, num_spans, num_head, head_hidden_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)  # type: ignore
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class LayoutSelfOutput(nn.Module):
    def __init__(self, config: LayoutModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutAttention(nn.Module):
    def __init__(self, config: LayoutModelConfig):
        super().__init__()
        self.self = LayoutSelfAttention(config)
        self.output = LayoutSelfOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        rel_2d_pos,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            rel_2d_pos,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class LayoutIntermediate(nn.Module):
    def __init__(self, config: LayoutModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LayoutOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutLayer(nn.Module):
    def __init__(self, config: LayoutModelConfig):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = LayoutAttention(config)
        self.intermediate = LayoutIntermediate(config)
        self.output = LayoutOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        rel_2d_pos,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            rel_2d_pos,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        return outputs


def relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1)
    )

    ret += torch.where(is_small, n, val_if_large)  # type: ignore
    return ret


class LayoutEncoder(nn.Module):
    def __init__(self, config: LayoutModelConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [LayoutLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.max_rel_2d_pos = config.max_rel_2d_pos
        self.rel_2d_pos_bins = config.real_2d_num_buckets
        self.rel_2d_pos_onehot_size = config.real_2d_num_buckets
        self.rel_pos_x_bias = nn.Linear(
            self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False
        )
        self.rel_pos_y_bias = nn.Linear(
            self.rel_2d_pos_onehot_size, config.num_attention_heads, bias=False
        )

    def _calculate_2d_position_embeddings(self, bbox):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(
            -1
        )
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(
            -1
        )
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_x = F.one_hot(rel_pos_x, num_classes=self.rel_2d_pos_onehot_size)
        rel_pos_y = F.one_hot(rel_pos_y, num_classes=self.rel_2d_pos_onehot_size)
        rel_pos_x = self.rel_pos_x_bias(rel_pos_x).permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias(rel_pos_y).permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        bbox,
        output_attentions=False,
    ):
        all_self_attentions = () if output_attentions else None
        rel_2d_pos = self._calculate_2d_position_embeddings(bbox)

        for _, layer_module in enumerate(self.layer):

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                rel_2d_pos,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if all_self_attentions is not None:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if all_self_attentions is not None:
            return (hidden_states, all_self_attentions)
        else:
            return (hidden_states,)


class LayoutModel(nn.Module):
    def __init__(self, config: LayoutModelConfig):
        super().__init__()
        self.config = config
        self.embeddings = LayoutEmbeeding(config)
        self.encoder = LayoutEncoder(config)

        for param in self.parameters():
            self.init_weights(param)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def calc_layout_embeddings(self, bbox):
        # span_semantic_embeddings: (batch_size, num_spans, hidden_size)
        # position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings.calc_layout_embedding(bbox)
        # embeddings = span_semantic_embeddings + position_embeddings + spatial_position_embeddings + style_embeddings
        embeddings = self.embeddings.LayerNorm(spatial_position_embeddings)
        embeddings = self.embeddings.dropout(spatial_position_embeddings)
        return embeddings

    def forward(
        self,
        bbox,
        attention_mask=None,
        position_ids=None,
        output_attentions: bool = False,
    ):
        device = bbox.device
        input_shape = bbox.shape[:-1]
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = -extended_attention_mask + 1

        if position_ids is None:
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.expand(input_shape)

        layout_hidden_states = self.calc_layout_embeddings(
            bbox=bbox,
        )

        encoder_outputs = self.encoder(
            layout_hidden_states, extended_attention_mask, bbox, output_attentions
        )
        return encoder_outputs
