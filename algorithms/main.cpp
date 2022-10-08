//
// Created by zy on 2022/9/24.
//
#include <bits/stdc++.h>
#include "C1-Basic-concepts/algorithm-1.1.h"
#include "C1-Basic-concepts/mathematical_preliminaries_1_2.h"

using namespace std;

int main() {
    int m, n;
    cin >> m >> n;
    cout << Euclid_algorithm(m, n) << endl;
    auto ex = Extended_Euclids_algorithm(m, n);
    for (auto it = ex.begin(); it != ex.end(); it++) {
        if (it == ex.end() - 1)
            cout << *it << endl;
        else
            cout << *it << " ";
    }
}