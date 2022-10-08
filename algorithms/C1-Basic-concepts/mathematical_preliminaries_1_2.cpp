//
// Created by zy on 2022/10/8.
//

#include "mathematical_preliminaries_1_2.h"

std::vector<int> Extended_Euclids_algorithm(int m, int n) {
    int a1 = 1, b = 1;
    int a = 0, b1 = 0;
    if (n > m)
        std::swap(m, n);
    int c = m, d = n;
    int q = c / d, r = c % d;
    int t;
    while (r != 0) {
        c = d;
        d = r;
        t = a1;
        a1 = a;
        a = t - q * a;
        t = b1;
        b1 = b;
        b = t - q * b;
        q = c / d;
        r = c % d;
    }
    return {a, b, d};
}