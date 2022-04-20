from typing import Optional

import torch
import torch.nn as nn

from transformers.models.gpt2 import GPT2LMHeadModel
from coca.mapper_model import LinearMapper


class CaptionModel(nn.Module):

    def __init__(self, num_prefix: int, frozen: bool = True):
        super().__init__()
        self.num_prefix = num_prefix
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        hidden_size = self.gpt.transformer.wte.weight.shape[1]
        self.mapper = LinearMapper(512, num_prefix, prefix_hidden_size=hidden_size)
        if frozen:
            for paramters in self.gpt.parameters():
                paramters.requires_grad_(False)

    def get_dummy_labels(self, batch_size: int, device):
        return torch.full((batch_size, self.num_prefix), fill_value=-100, device=device)

    def get_dummy_attention_mask(self, batch_size: int, device):
        return torch.ones((batch_size, self.num_prefix), device=device)

    def forward(self, input_ids: torch.Tensor, image_feature: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, return_loss: bool=True):
        batch_size = input_ids.size(0)
        device = input_ids.device

        text_embedding = self.gpt.transformer.wte(input_ids)
        prefix_embedding = self.mapper(image_feature)
        embedding = torch.cat([prefix_embedding, text_embedding], dim=1)
        if attention_mask is not None:
            dummy_attention_mask = self.get_dummy_attention_mask(batch_size, device)
            attention_mask = torch.cat((dummy_attention_mask, attention_mask), dim=1)
        if return_loss:
            dummy_labels = self.get_dummy_labels(batch_size, device)
            labels = torch.cat((dummy_labels, input_ids), dim=1)
        else:
            labels = None
        out = self.gpt(inputs_embeds=embedding, labels=labels, attention_mask=attention_mask)
        return out