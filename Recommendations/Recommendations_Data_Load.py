#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : Recommendations_Data_Load.py
@ide    : PyCharm
@time   : 2022/8/29 17:03
"""
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


class Load_titanic_train_data:
    def __init__(self, path: str):
        self._path = path
        self._df_data = pd.read_csv(path)
        self.sparse_feature_list = ["Pclass", "Sex", "Cabin", "Embarked"]
        self.dense_feature_list = ["Age", "SibSp", "Parch", "Fare"]
        self.__recodedAndfillnan()
        self._split()

    def __recodedAndfillnan(self):
        sparse_feature_reindex_dict = {}
        for i in self.sparse_feature_list:
            cur_sparse_feature_list = self._df_data[i].unique()

            sparse_feature_reindex_dict[i] = dict(zip(cur_sparse_feature_list,
                                                      range(1, len(cur_sparse_feature_list) + 1)
                                                      )
                                                  )

            self._df_data[i] = self._df_data[i].map(sparse_feature_reindex_dict[i])

        for j in self.dense_feature_list:
            self._df_data[j] = self._df_data[j].fillna(0)

    def _split(self):
        data = self._df_data[self.sparse_feature_list + self.dense_feature_list]
        label = self._df_data["Survived"].values
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(data, label, test_size=0.2,
                                                                            random_state=2020)

    def getSplitdata(self):
        return self.xtrain, self.xtest, self.ytrain, self.ytest


class MovieLens1MDataset(Dataset):
    """
    MovieLens 20M Dataset

    Data preparation
        treat samples with a rating less than 3 as negative samples

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep='::', engine='python', header=None):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int) - 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0,), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

class MovieLens20MDataset(MovieLens1MDataset):
    def __init__(self,dataset_path):
        super(MovieLens20MDataset, self).__init__(dataset_path,sep=',',engine='c',header='infer')
