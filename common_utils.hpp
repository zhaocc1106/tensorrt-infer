//
// Created by zhaocc on 2021/7/15.
//

#ifndef TRT_TEST__COMMON_UTILS_HPP_
#define TRT_TEST__COMMON_UTILS_HPP_

#include <memory>
#include "NvInfer.h"

class MyLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) {
    if (severity <= nvinfer1::ILogger::Severity::kWARNING) {
      std::cout << msg << std::endl;
    }
  }
};

template<typename T>
struct TrtDestroyer {
  void operator()(T* t) { t->destroy(); }
};

template<typename T> using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T> >;

#endif //TRT_TEST__COMMON_UTILS_HPP_
