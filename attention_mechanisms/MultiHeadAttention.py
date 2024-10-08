import math

import torch
import torch.nn as nn
from torch.nn import Module


####################################################################################################
# Multi-Head Attention Layer
# The Multi-Head Attention layer is a modified version of the one used in the Transformer architecture.
# The original implementation can be found here: 
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention
####################################################################################################
class MultiHeadAttention(Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.attention_head_size = int(in_channels / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

        self.query = nn.Linear(in_channels, self.all_head_size)
        self.key = nn.Linear(in_channels, self.all_head_size)
        self.value = nn.Linear(in_channels, self.all_head_size)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(in_channels, in_channels)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)

        return attention_output

        # # Apply multi-head attention at a suitable position
        # if idx == 5:
        #     #print(f"Before MultiHeadAttention, x shape: {x.shape}")

        #     # Reshape x for MultiHeadAttention
        #     batch_size, channels, height, width = x.shape
        #     # Reshaping to [batch_size, seq_len, feature_dim]
        #     x = x.view(batch_size, height * width, channels)

        #     # Apply MultiHeadAttention
        #     x = self.multi_head_attention(x)

        #     # Reshape back to original dimensions if needed
        #     x = x.view(batch_size, channels, height, width)
