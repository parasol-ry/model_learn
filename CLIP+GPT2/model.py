from distutils.command.config import config
import math
from turtle import forward
from typing import Iterator, Optional
from unittest import result
from pydantic import NoneBytes
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import GPT2Tokenizer
from transformers.models.gpt2 import GPT2LMHeadModel 
from torch.nn.functional import relu, softmax, cross_entropy
from torch.nn import CrossEntropyLoss
from typing import Tuple, Optional, Union


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, num_heads, bias = True, dropout = 0.) -> None:
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.attention_head_size = int(dim_self / num_heads)
        self.all_head_size = self.attention_head_size * self.num_heads

        self.query = nn.Linear(dim_self, self.all_head_size)
        self.key = nn.Linear(dim_self, self.all_head_size)
        self.value = nn.Linear(dim_self, self.all_head_size)

        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(dim_self, dim_self)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, y = None, mask = None):
        mixed_query_layer = self.query(x) # (batch_size, seq_len, embed_dim)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        #caluate q * k(T)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if mask is not None:
            attention_scores = attention_scores + mask

        attention_probs = softmax(attention_scores, dim = -1)
        attention_probs = self.dropout(attention_probs)

        contex_layer = torch.matmul(attention_probs, value_layer)
        contex_layer = contex_layer.permute(0, 2, 1, 3).contiguous()
        new_contex_layer_shape = contex_layer.size()[:-2] + (self.all_head_size,)
        contex_layer = contex_layer.view(new_contex_layer_shape)

        out = self.project(contex_layer)

        return (out,)

class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act = relu, dropout = 0.) -> None:
        super(MlpTransformer, self).__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class TransformerLayer(nn.Module):
    def __init__(self, dim_self, num_heads, mlp_ratio = 4., bias = False, dropout = 0., act = relu, 
                norm_layer: nn.Module = nn.LayerNorm) -> None:
        super(TransformerLayer, self).__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, num_heads, bias = bias, dropout = dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act = act, dropout = dropout)
    
    def forward(self, x, y = None, mask = None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                mlp_ratio: float = 2., act = relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False) -> None:
        super(Transformer, self).__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(TransformerLayer(dim_self, num_heads, mlp_ratio, act = act, norm_layer = norm_layer))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, y = None, mask = None):
        for i, layer in enumerate(self.layers):
            x = layer(x, y, mask)
        return x

class TransformerMapper(nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8) -> None:
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length*dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad = True)
        
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim = 1)
        out = self.transformer(prefix)
        out = out[:, self.clip_length:]
        return out


class ClipGPT2Model(nn.Module):
    def __init__(self, prefix_length: int, clip_length: int, prefix_size: int = 512, 
                num_layers: int = 8) -> None:
        super(ClipGPT2Model, self).__init__()
        self.preifx_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length, 
        clip_length, num_layers)
        # self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
        #                              self.gpt_embedding_size * prefix_length))
        # self.clip_project = LinearMapper(512, 8, prefix_hidden_size=768)

        #冻结一部分参数
        for paramters in self.gpt.parameters():
            paramters.requires_grad_(False)

        
    def get_dummy_labels(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size, self.preifx_length), fill_value = -100, device = device)

    def get_dummy_attention_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, self.preifx_length), device = device)

    def forward(self, input_ids: torch.Tensor, image_feature: torch.Tensor, attention_mask: torch.Tensor, 
                return_loss: bool=True):
        embedding_text = self.gpt.transformer.wte(input_ids)
        prefix_projections = self.clip_project(image_feature)
        # prefix_projections = self.clip_project(image_feature).view(-1, self.preifx_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim = 1)
        
        # with torch.no_grad():
        
        
        if return_loss:
            dummy_labels = self.get_dummy_labels(input_ids.shape[0], input_ids.device)
            
            mask_labels = (1 - attention_mask)*(-100)
            labels = input_ids * attention_mask + mask_labels
            labels = torch.cat((dummy_labels, labels), dim = 1)

        else :
            labels = None
        # print(labels)
        if attention_mask is not None:
            dummy_attention_mask = self.get_dummy_attention_mask(input_ids.shape[0], input_ids.device)
            attention_mask = torch.cat((dummy_attention_mask, attention_mask), dim = 1)

        out = self.gpt(inputs_embeds = embedding_cat, labels = labels, attention_mask = attention_mask)

        return out

    # def train(self, mode: bool = True):
    #     super(ClipGPT2Model, self).train(mode)
    #     self.gpt.eval()
    #     return self

    # def parameters(self, recurse: bool = True):
    #     return self.clip_project.parameters()
    
    # def train(self, mode: bool = True):
    #     super(ClipGPT2Model, self).train(mode)
    #     self.gpt.eval()
    #     return self

