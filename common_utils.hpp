//
// Created by zhaocc on 2021/7/15.
//

#ifndef TRT_TEST__COMMON_UTILS_HPP_
#define TRT_TEST__COMMON_UTILS_HPP_

#include <memory>
#include <ctime>
#include <cstring>
#include <sys/time.h>
#include <cmath>
#include <random>

#include "NvInfer.h"

class MyLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) {
    if (severity <= nvinfer1::ILogger::Severity::kINFO) {
      /*Generate time str such as 2017-08-05 09:22:55.726*/
      timeval tv{};
      gettimeofday(&tv, nullptr);
      struct tm* tm_now = localtime(&tv.tv_sec);
      char time_str[30];
      memset(time_str, 0, sizeof(time_str));
      strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_now);
      snprintf(time_str + 19, 5, ".%lu", (unsigned long) round(tv.tv_usec / 1000.0));

      std::string severity_flag;
      switch (severity) {
        case nvinfer1::ILogger::Severity::kVERBOSE:severity_flag = "[V]";
          break;
        case nvinfer1::ILogger::Severity::kINFO:severity_flag = "[I]";
          break;
        case nvinfer1::ILogger::Severity::kWARNING:severity_flag = "[W]";
          break;
        case nvinfer1::ILogger::Severity::kERROR:severity_flag = "[E]";
          break;
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:severity_flag = "[IE]";
          break;
      }

      std::cout << "[" << time_str << "] " << severity_flag << " " << msg << std::endl;
    }
  }
};

template<typename T>
struct TrtDestroyer {
  void operator()(T* t) { t->destroy(); }
};

template<typename T> using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T> >;

template<typename T>
inline T RoundUp(T m, T n) { return ((m + n - 1) / n) * n; }

/**
 * Calc volume of tensor.
 * @param d: Dimensions of tensor.
 * @return The volume of tensor.
 */
inline int Volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>()); // 所有维度shape乘积
}

/**
 * Calc volume of tensor.
 * @param dims: Dimensions of tensor.
 * @param vec_dim:
 * @param comps
 * @param batch: Batch size.
 * @return The volume of tensor.
 */
inline int Volume(nvinfer1::Dims dims, int vec_dim, int comps, int batch = 1) {
  if (vec_dim != -1) {
    dims.d[vec_dim] = RoundUp(dims.d[vec_dim], comps);
  }
  return Volume(dims) * std::max(batch, 1);
}

/**
 * Fill buffer with uniform random.
 * @param buffer: Buffer.
 * @param volume: Buffer volume.
 * @param min: Min val of uniform distribution.
 * @param max: Max val of uniform distribution.
 */
template<typename T>
inline void FillBuffer(void* buffer, int volume, T min, T max) {
  T* typed_buffer = static_cast<T*>(buffer);
  std::default_random_engine engine;
  if (std::is_integral<T>::value) {
    std::uniform_int_distribution<int> distribution(min, max);
    auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
    std::generate(typed_buffer, typed_buffer + volume, generator);
  } else {
    std::uniform_real_distribution<float> distribution(min, max);
    auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
    std::generate(typed_buffer, typed_buffer + volume, generator);
  }
}

#endif //TRT_TEST__COMMON_UTILS_HPP_
