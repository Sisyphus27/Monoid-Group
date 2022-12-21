//
// Created by zy on 2022/12/21.
//
#include "Ssort.h"
#include "utils.h"

int main() {
    std::vector<int> arr = {1, 3, 4, 6, 1, 4, 22, 5, 25};
    Sprint(arr);
    s_insertion_sort(arr);
    Sprint(arr);
    return 0;
}