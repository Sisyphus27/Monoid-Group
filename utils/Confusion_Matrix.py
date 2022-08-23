#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file   : Confusion_Matrix.py
@ide    : PyCharm
@time   : 2022/8/23 15:02
"""
import os

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MixMatrixStatic:
    """
    description:
        for binary classify

    input:
        predict vector : ndarray
        truth vector : ndarray
        threshold : float

    method:
        draw() : it will output a picture of confusion matrix
        statistics() : it will return three values p,r,f1
        roc() : return three vector and auc value
        predict() : return a vector with "Yes" or "No"
    """

    def __init__(self, pred_vector: np.ndarray, y_vector: np.ndarray = None, threshold: float = 0.5,
                 model: str = ""):
        """

        """
        self._mode = model
        self._ori = pred_vector.copy()
        self._pred = pred_vector
        self._y = y_vector
        self._threshold = threshold
        self.__normalized()
        # self.fp1 = self.__fp()
        # self.tp1 = self.__tp()
        # self.fn1 = self.__fn()
        # self.tn1 = self.__tn()
        if self._mode == "with_label":
            self.__static()

    def __static(self):
        try:
            self.__C = confusion_matrix(self._y, self._pred)
        except ValueError:
            print("Please put label before call this method")
            quit()
        self.__total = self.__C.sum()
        self.__tn = self.__C[0][0]
        self.__fn = self.__C[1][0]
        self.__fp = self.__C[0][1]
        self.__tp = self.__C[1][1]
        self.__p = self.__tp / (self.__tp + self.__fp)
        self.__r = self.__tp / (self.__tp + self.__fn)
        self.__s = self.__tn / (self.__tn + self.__fp)
        self.__f1 = (2 * self.__p * self.__r) / (self.__r + self.__p)

    def __normalized(self):
        indices0 = self._ori <= self._threshold
        indices1 = self._ori > self._threshold
        self._pred[indices0] = 0
        self._pred[indices1] = 1
        return

    def statistics(self):
        if self._mode != "with_label":
            print("Set mode is 'with_label' firstly : statistics()")
            quit()
        print('p: {:2f}'.format(self.__p))
        print('r: {:2f}'.format(self.__r))
        print('s: {:2f}'.format(self.__s))
        print('f1: {:2f}'.format(self.__f1))
        return self.__p, self.__r, self.__s, self.__f1

    def draw(self):
        if self._mode != "with_label":
            print("Set mode is with_label firstly : draw()")
            quit()
        # This parameter used for cancer/disease classifier
        char_type = True
        if char_type == False:
            hotmap_data = pd.DataFrame(columns=('pre_true', 'pre_false'), index=('ground_true', 'ground_false'))
            hotmap_data['pre_true']['ground_true'] = self.__tp
            hotmap_data['pre_false']['ground_true'] = self.__fn
            hotmap_data['pre_true']['ground_false'] = self.__fp
            hotmap_data['pre_false']['ground_false'] = self.__tn
            hotmap_data = hotmap_data.astype(float)
            ax = sns.heatmap(data=hotmap_data, square=True, annot=True, fmt='.20g')
            plt.show()
        else:
            hotmap_data = pd.DataFrame(columns=('pre_cancer', 'pre_disease'), index=('ground_cancer', 'ground_disease'))
            hotmap_data['pre_cancer']['ground_cancer'] = self.__tp
            hotmap_data['pre_disease']['ground_cancer'] = self.__fn
            hotmap_data['pre_cancer']['ground_disease'] = self.__fp
            hotmap_data['pre_disease']['ground_disease'] = self.__tn
            hotmap_data = hotmap_data.astype(float)
            ax = sns.heatmap(data=hotmap_data, square=True, annot=True, fmt='.20g')
            plt.show()

    def roc(self, print_set=True):
        if self._mode != "with_label":
            print("Set mode is with_label firstly : roc()")
            quit()
        # fpr_test, tpr_test, th_test = roc_curve(self._y, self._pred)
        # self.__auc = auc(fpr_test, tpr_test)
        self.__auc = roc_auc_score(self._y, self._ori)
        if print_set is True:
            print('auc: {:2f}'.format(self.__auc))
        # return fpr_test, tpr_test, th_test, self.__auc
        return self.__auc

    def predict(self):
        result = []
        for x in self._pred:
            if x == 1:
                result.append("Yes")
            else:
                result.append("No")
        return result

    def draw_all(self):
        try:
            os.mkdir('output')
        except FileExistsError:
            print('file has been create.')
        os.chdir('./output')
        therhold = {0.1, 0.2, 0.3, 0.4, 0.5}
        for i in therhold:
            self._threshold = i
            self.__normalized()
            if self._mode == "with_label":
                self.__static()
            if self._mode != "with_label":
                print("Set mode is with_label firstly : draw()")
                quit()
            # This parameter used for cancer/disease classifier
            char_type = True
            if char_type == False:
                plt.figure('{}.png'.format(i))
                hotmap_data = pd.DataFrame(columns=('pre_true', 'pre_false'), index=('ground_true', 'ground_false'))
                hotmap_data['pre_true']['ground_true'] = self.__tp
                hotmap_data['pre_false']['ground_true'] = self.__fn
                hotmap_data['pre_true']['ground_false'] = self.__fp
                hotmap_data['pre_false']['ground_false'] = self.__tn
                hotmap_data = hotmap_data.astype(float)
                ax = sns.heatmap(data=hotmap_data, square=True, annot=True, fmt='.20g')
            else:
                plt.figure('{}.png'.format(i))
                hotmap_data = pd.DataFrame(columns=('pre_cancer', 'pre_disease'),
                                           index=('ground_cancer', 'ground_disease'))
                hotmap_data['pre_cancer']['ground_cancer'] = self.__tp
                hotmap_data['pre_disease']['ground_cancer'] = self.__fn
                hotmap_data['pre_cancer']['ground_disease'] = self.__fp
                hotmap_data['pre_disease']['ground_disease'] = self.__tn
                hotmap_data = hotmap_data.astype(float)
                ax = sns.heatmap(data=hotmap_data, square=True, annot=True, fmt='.20g')
            plt.savefig('{}.png'.format(i))
            file_path = str(i) + '.txt'
            f = open(file_path, 'w')
            f.write('p: {:2f}'.format(self.__p) + '\n')
            f.write('r: {:2f}'.format(self.__r) + '\n')
            f.write('s: {:2f}'.format(self.__s) + '\n')
            f.write('f1: {:2f}'.format(self.__f1) + '\n')
            try:
                self.roc(print_set=False)
            except NotImplementedError:
                print('roc()')
            f.write('auc: {:2f}'.format(self.__auc) + '\n')
            f.close()
