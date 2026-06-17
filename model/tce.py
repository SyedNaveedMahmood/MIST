# Faithful port of AttnSleep's Temporal Context Encoder (TCE):
# multi-head attention with causal convolutions + position-wise feed-forward,
# stacked N times with residual + layer norm. Reference: Eldele et al., 2021.
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True):
        self._pad = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=self._pad, dilation=dilation,
                         groups=groups, bias=bias)

    def forward(self, x):
        out = super().forward(x)
        if self._pad != 0:
            return out[:, :, :-self._pad]
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.convs = clones(
            CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size,
                         kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        # (B, afr, d_model) -> causal conv -> split heads
        query = self.convs[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        x, _ = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.sublayer_output[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer_output[1](x, self.feed_forward)


class TCE(nn.Module):
    """Stack of N encoder layers with a final layer norm."""

    def __init__(self, layer, n):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def build_tce(d_model=80, d_ff=120, h=5, n=2, afr_reduced_cnn_size=30, dropout=0.1):
    attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff),
                         afr_reduced_cnn_size, dropout)
    return TCE(layer, n)
