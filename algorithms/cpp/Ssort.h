//
// Created by zy on 2023/1/12.
//

#ifndef CPP_SSORT_H
#define CPP_SSORT_H

#include <utility>
#include <vector>
#include <iostream>
#include "utils.h"

void insertion_sort(std::vector<int> &arr);

class sHeap {
public:
    sHeap(std::vector<int> arr) : heap(std::move(arr)) {
        sPrint(arr);
        this->size = this->heap.size();
        _buildmaxheap(size);
    };

    void showHeap();

    void showSize() const;

    std::vector<int>heapsort();

private:
    std::vector<int> heap;
    int size;
    bool maxHeap = true;

    static int _parent(int idx);

    static int _left(int idx);

    static int _right(int idx);

    void _maxheapify(int idx);

    void _buildmaxheap(int idx);

    std::vector<int> _heapsort(int idx);
};

#endif //CPP_SSORT_H
