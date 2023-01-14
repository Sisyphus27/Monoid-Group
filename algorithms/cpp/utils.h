//
// Created by zy on 2023/1/15.
//

#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <vector>
#include <iostream>

template<class T>
void sPrint(std::vector<T> &arr) {
    for (int i = 0; i < arr.size(); ++i) {
        if (i != arr.size() - 1)
            std::cout << arr[i] << " ";
        else
            std::cout << arr[i] << std::endl;
    }
}

template<class T>
void sPrint(T num) {
    std::cout << num << std::endl;
}


#endif //CPP_UTILS_H
