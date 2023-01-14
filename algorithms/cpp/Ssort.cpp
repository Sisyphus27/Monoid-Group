//
// Created by zy on 2023/1/12.
//

#include "Ssort.h"


void insertion_sort(std::vector<int> &arr) {
    if (arr.size() <= 1)
        return;
    for (int i = 0; i < arr.size(); ++i) {
        int key = arr[i];
        int j = i - 1;
        while (j > 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

int sHeap::_parent(int idx) {
    return (idx - 1) >> 1;
}

int sHeap::_left(int idx) {
    return (idx << 1) + 1;
}

int sHeap::_right(int idx) {
    return (idx << 1) + 2;
}

void sHeap::showHeap() {
    sPrint(this->heap);
}

void sHeap::showSize() const {
    sPrint(this->size);
}

void sHeap::_maxheapify(int idx) {
    int l = _left(idx);
    int r = _right(idx);
//    std::cout << idx << " " << this->heap[idx] << std::endl;
//    std::cout << l << " " << this->heap[l] << std::endl;
//    std::cout << r << " " << this->heap[r] << std::endl;
    int largest;
    if (l <= this->size - 1 && this->heap[l] > this->heap[idx])
        largest = l;
    else
        largest = idx;
    if (r <= this->size - 1 && this->heap[r] > this->heap[largest])
        largest = r;
    if (largest != idx) {
        std::swap(this->heap[largest], this->heap[idx]);
        _maxheapify(largest);
    }
}

void sHeap::_buildmaxheap(int idx) {
    this->size = idx;
    for (int i = idx >> 1; i >= 0; --i) {
        _maxheapify(i);
    }
}

std::vector<int> sHeap::_heapsort(int idx) {
    _buildmaxheap(idx);
    for (int i = idx - 1; i >= 0; --i) {
        std::swap(this->heap[0], this->heap[i]);
        this->size--;
        _maxheapify(0);
    }
    return std::move(heap);
}

std::vector<int> sHeap::heapsort() {
    return _heapsort(this->size);
}

