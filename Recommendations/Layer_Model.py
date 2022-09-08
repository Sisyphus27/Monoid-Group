#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : Layer_Model.py
@ide    : PyCharm
@time   : 2022/9/5 15:21
"""
import numpy as np
import torch
import torch.nn as nn


class FeaturesLinear(nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]),
                                dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ''(batch_size, num_fields)''
        :return:
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ''(batch_size, num_fields)''
        :return:
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ''(batch_size, num_fields, embed_dim)''
        :return:
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        # self.linear=FFM_FeatureLinear(field_dims=field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array(0, *np.cumsum([field_dims][:-1]), dtype=np.long)
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ''(batch_size, num_fields)''
        :return:
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ''(batch_size, embed_dim)''
        :return:
        """
        return self.mlp(x)


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.w = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(self.num_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros((input_dim,))) for _ in range(self.num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return:
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x
