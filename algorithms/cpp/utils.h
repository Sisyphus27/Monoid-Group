//
// Created by zy on 2022/12/21.
//

#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <iostream>
#include <vector>

template<class T>
void Sprint(std::vector<T> &arr) {
    for (int i = 0; i < arr.size(); ++i) {
        if (i != arr.size() - 1)
            std::cout << arr[i] << " ";
        else
            std::cout << arr[i] << std::endl;
    }
}

#endif //CPP_UTILS_H
