#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file    : Ssort
@ide     : PyCharm
@time    : 2022/12/21 15:25
"""
import math


def _insertion_sort_s(arr):
    if len(arr) <= 1:
        return
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j > 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

    return


def _merge_s(arr, p, q, r):
    nL = q - p + 1
    nR = r - q
    L = []
    R = []
    for i in range(nL):
        L.append(arr[p + i])
    for i in range(nR):
        R.append(arr[q + i + 1])
    i, j, k = 0, 0, p
    while i < nL and j < nR:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    while i < nL:
        arr[k] = L[i]
        i += 1
        k += 1
    while j < nR:
        arr[k] = R[j]
        j += 1
        k += 1


# 算法导论 p39
def _merge_sort_s(arr, p, r):
    if p >= r:
        return
    q = math.floor((p + r) / 2)
    _merge_sort_s(arr, p, q)
    _merge_sort_s(arr, q + 1, r)
    _merge_s(arr, p, q, r)


arr = [1, 3, 4, 5, 6, 53, 6, 43, 64, 5]

_merge_sort_s(arr, 0, len(arr) - 1)
print(0)
