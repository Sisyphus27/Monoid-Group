//
// Created by zy on 2022/12/21.
//
#include <vector>
#include <algorithm>
#include "Ssort.h"
#include "utils.h"

int main() {
    std::vector<int> s = {5,13,2,25,7,17,20,8,4};
    insertion_sort(s);
    auto k=sHeap(s);
    sPrint(s);
    k.showHeap();
    k.showSize();
    auto o=k.heapsort();
    sPrint(o);
    return 0;
}