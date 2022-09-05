#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : Field_aware_Factorization_Machines_Model.py
@ide    : PyCharm
@time   : 2022/8/31 11:18
"""
import numpy as np
import torch
from Recommendations.Layer_Model import FieldAwareFactorizationMachine, FeaturesLinear

import torch.nn as nn


class FieldAwareFactorizationMachineModel(nn.Module):
    """
    A pytorch implementation of Field-aware Factorization Machine.

    Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim):
        super(FieldAwareFactorizationMachineModel, self).__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return:
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))
