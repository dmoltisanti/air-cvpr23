# copied from Action Modifier: https://github.com/hazeld/action-modifiers/blob/master/model.py


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, emb_dim, heads=1, dropout=0.1):
        super(SDPAttention, self).__init__()
        self.d_k = int(d_k/heads)
        self.d_v = int(d_v/heads)
        d_model = int(d_model/heads)
        self.q_linear = nn.Linear(int(emb_dim/heads), self.d_k)
        self.k_linear = nn.Linear(d_model, self.d_k)
        self.v_linear = nn.Linear(d_model, self.d_v)
        self.h = heads

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(heads*self.d_v, emb_dim)

    def attention(self, q, k, v, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        output = output.reshape(output.shape[0], output.shape[1], output.shape[-1])

        return output, scores

    def forward(self, features, queries):
        bs = queries.shape[0]
        q = self.q_linear(queries.view(bs, -1, self.h, int(queries.shape[-1]/self.h)))
        k = self.k_linear(features.view(bs, -1, self.h, int(features.shape[-1]/self.h)))
        v = self.v_linear(features.view(bs, -1, self.h, int(features.shape[-1]/self.h)))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output, scores = self.attention(q, k, v, self.dropout)
        concat = output.transpose(1, 2).contiguous().view(bs, self.d_v*self.h)
        output = self.out(concat)

        return [output, scores]
