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
import statistics


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


class sHeap:
    def __init__(self, arr: list):
        self._heap = arr.copy()
        self._size = len(self._heap)
        self._maxHeap = True
        self.__buildmaxheap(self._size)

    @staticmethod
    def _parent(idx):
        return (idx - 1) >> 1

    @staticmethod
    def _left(idx):
        return (idx << 1) + 1

    @staticmethod
    def _right(idx):
        return (idx << 1) + 2

    def __maxheapify(self, idx):
        l = self._left(idx)
        r = self._right(idx)
        if l <= self._size - 1 and self._heap[l] > self._heap[idx]:
            largest = l
        else:
            largest = idx
        if r <= self._size - 1 and self._heap[r] > self._heap[largest]:
            largest = r
        if largest != idx:
            self._heap[idx], self._heap[largest] = self._heap[largest], self._heap[idx]
            self.__maxheapify(largest)
        return

    def __buildmaxheap(self, n):
        self._size = n
        for i in range(math.floor(self._size / 2), -1, -1):
            if self._maxHeap:
                self.__maxheapify(i)

    def heapsort(self):
        return self.__heapsort(self._size)

    def __heapsort(self, idx):
        self.__buildmaxheap(idx)
        for i in range(idx - 1, -1, -1):
            self._heap[0], self._heap[i] = self._heap[i], self._heap[0]
            self._size -= 1
            self.__maxheapify(0)
        return self._heap


arr = [1, 3, 4, 5, 6, 53, 6, 43, 64, 5]

# _merge_sort_s(arr, 0, len(arr) - 1)
j = sHeap(arr)
aij = j.heapsort()
print(0)
