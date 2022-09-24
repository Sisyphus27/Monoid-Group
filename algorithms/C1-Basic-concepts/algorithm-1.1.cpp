//
// Created by zy on 2022/9/24.
//

#include "algorithm-1.1.h"

int Euclid_algorithm(int m, int n) {
    if (n > m)
        std::swap(m, n);
    int r = m % n;
    while (r != 0) {
        m = n;
        n = r;
        r = m % n;
    }
    return n;
}