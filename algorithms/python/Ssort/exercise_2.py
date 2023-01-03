#!/user/bin/env python
# coding=utf-8
"""
@project : Monoid-Group
@author  : zy
@file    : exercise_2
@ide     : PyCharm
@time    : 2022/12/24 10:44
"""

import random


# 2.3-5
def _recursive_insertion_sort(arr: list, n):
    if n <= 1:
        return
    _recursive_insertion_sort(arr, n - 1)
    last_element = arr[n - 1]
    next = n - 2
    while n > 0 and arr[next] > last_element:
        arr[next + 1] = arr[next]
        next -= 1
    arr[next + 1] = last_element


arr = [1, 5, 3, 1, 4, 5, 2, 5, 6, 3, 46, 1, 4]
_recursive_insertion_sort(arr, len(arr))
print(0)
