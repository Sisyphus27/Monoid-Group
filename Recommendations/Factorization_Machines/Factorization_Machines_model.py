#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : Factorization_Machines_model.py
@ide    : PyCharm
@time   : 2022/8/23 15:37
"""
import numpy as np
import torch
import torch.nn as nn
from Recommendations.Layer_Model import FeaturesLinear,FeaturesEmbedding,FactorizationMachine



class FactorizationMachineModel(nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    """

    def __init__(self, field_dims, embed_dim):
        super(FactorizationMachineModel, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ''(batch_size, num_fields)''
        :return:
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
