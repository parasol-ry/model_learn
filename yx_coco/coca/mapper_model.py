import torch
import torch.nn as nn


class LinearMapper(nn.Module):

    def __init__(self, image_feature_size: int, num_prefix: int, prefix_hidden_size: int):
        super().__init__()
        self.image_feature_size = image_feature_size
        self.num_prefix = num_prefix
        self.prefix_hidden_size = prefix_hidden_size

        second_size = num_prefix * prefix_hidden_size // 2
        self.linear = nn.Sequential(
            nn.Linear(image_feature_size, second_size),
            nn.Tanh(),
            nn.Linear(second_size, num_prefix * prefix_hidden_size),
        )
    
    def forward(self, image_feature: torch.Tensor):
        # image_feature: (batch_size, image_feature_size)
        prefix_size = (self.num_prefix, self.prefix_hidden_size)
        # preix: (batch_size, num_prefix, prefix_hidden_size)
        prefix_embedding = self.linear(image_feature).view(-1, *prefix_size)
        return prefix_embedding
