#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : FM.py
@ide    : PyCharm
@time   : 2022/8/23 15:37
"""
from Factorization_Machines_model import FactorizationMachineModel
from Recommendations.Recommendations_Data_Load import Load_titanic_train_data
from Recommendations.Recommendations_Data_Load import Load_unKnow_data
from Recommendations.Recommendations_Data_Load import Unknow_data_torch_load
import numpy as np

# a = [
#     [[0., 1., 6., 11.],
#      [1., 2., 3., 4.],
#      [4., 5., 6., 1.]]
# ]
# print(a)
# embedding_list = a[0]
# raw_fm_result = raw_fm_cross_layer(embedding_list)
# print(raw_fm_result)
#
# embedding_list=torch.tensor(a)
# fm_result=fm_cross_layer(embedding_list)
# print(fm_result)
dataSet = Load_titanic_train_data('Titanic-data/train.csv')
xtrain, xtest, ytrain, ytest = dataSet.getSplitdata()

xtrain_data = {"Pclass": np.array(xtrain["Pclass"]), \
               "Sex": np.array(xtrain["Sex"]), \
               "Cabin": np.array(xtrain["Cabin"]), \
               "Embarked": np.array(xtrain["Embarked"]), \
               "Age": np.array(xtrain["Age"]), \
               "SibSp": np.array(xtrain["SibSp"]), \
               "Parch": np.array(xtrain["Parch"]), \
               "Fare": np.array(xtrain["Fare"])}

xtest_data = {"Pclass": np.array(xtest["Pclass"]), \
              "Sex": np.array(xtest["Sex"]), \
              "Cabin": np.array(xtest["Cabin"]), \
              "Embarked": np.array(xtest["Embarked"]), \
              "Age": np.array(xtest["Age"]), \
              "SibSp": np.array(xtest["SibSp"]), \
              "Parch": np.array(xtest["Parch"]), \
              "Fare": np.array(xtest["Fare"])}

data=Load_unKnow_data('Unkown_data/train_data.txt')
feature,label=data.get_data()
DateSet=Unknow_data_torch_load(feature,label)
output=FactorizationMachineModel(DateSet)