//
// Created by zy on 2022/12/21.
//

#include "Ssort.h"

template<class T>
void s_insertion_sort(std::vector<T> &arr) {
    if (arr.size() <= 1)return;
    for (int i = 1; i < arr.size(); ++i) {
        T key = arr[i];
        int j = i - 1;
        while (j > 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}