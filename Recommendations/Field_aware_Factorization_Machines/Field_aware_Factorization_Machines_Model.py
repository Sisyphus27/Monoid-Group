#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : Field_aware_Factorization_Machines_Model.py
@ide    : PyCharm
@time   : 2022/8/31 11:18
"""
import torch

from Recommendations.Factorization_Machines.Factorization_Machines_model import \
    FeaturesEmbedding, FeaturesLinear
import torch.nn as nn


class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        input_dim = sum(field_dims)
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        # self.linear=FFM_FeatureLinear(field_dims=field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(input_dim + 1, embed_dim) for _ in range(self.num_fields)
        ])

    def forward(self, x):
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        ffm = torch.sum(ix, (1, 2))
        x = self.linear(x).squeeze(1) + ffm
        return x
