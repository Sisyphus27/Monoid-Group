#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : Deep_Cross_Network_Model.py
@ide    : PyCharm
@time   : 2022/9/7 10:17
"""
import torch
import torch.nn as nn
from Recommendations.Layer_Model import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron


class DeepCrossNetworkModel(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super(DeepCrossNetworkModel, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return:
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        x_l2 = self.mlp(embed_x)
        x_stack = torch.cat([x_l1, x_l2], dim=1)
        p = self.linear(x_stack)
        return torch.sigmoid(p.squeeze(1))
