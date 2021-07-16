//
// Created by zhaocc on 2021/7/15.
//

#ifndef TRT_TEST__UTILS_H_
#define TRT_TEST__UTILS_H_

#include <memory>

template<typename T>
struct TrtDestroyer {
  void operator()(T* t) { t->destroy(); }
};

template<typename T> using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T> >;

#endif //TRT_TEST__UTILS_H_
