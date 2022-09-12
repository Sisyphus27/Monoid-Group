#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : Neural_Factorization_Machines_Model.py
@ide    : PyCharm
@time   : 2022/9/12 14:55
"""
import torch.nn as nn
import torch
from Recommendations.Layer_Model import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear


class NeuralFactorizationMachineModel(nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.

    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super(NeuralFactorizationMachineModel, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return:
        """
        cross_term = self.fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term)
        return torch.sigmoid(x.squeeze(1))
