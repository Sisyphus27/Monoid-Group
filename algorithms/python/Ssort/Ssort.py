#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file    : Ssort
@ide     : PyCharm
@time    : 2022/12/21 15:25
"""


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


arr = [1, 3, 4, 5, 6, 53, 6, 43, 64, 5]

_insertion_sort_s(arr)
print(0)
