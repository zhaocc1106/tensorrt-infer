/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_TEST__TRT_CUDA_H
#define TRT_TEST__TRT_CUDA_H

#include <iostream>
#include <thread>
#include <utility>
#include <cuda_runtime.h>

inline void CudaCheck(cudaError_t ret, std::ostream& err = std::cerr) {
  if (ret != cudaSuccess) {
    err << "Cuda failure: " << cudaGetErrorString(ret) << std::endl;
    abort();
  }
}

class TrtCudaEvent;

#if CUDA_VERSION < 10000
void CudaSleep(cudaStream_t stream, cudaError_t status, void* sleep)
#else
void CudaSleep(void* sleep)
#endif
{
  std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(*static_cast<int*>(sleep)));
}

//!
//! \class TrtCudaStream
//! \brief Managed CUDA stream
//!
class TrtCudaStream {
 public:

  TrtCudaStream() {
    CudaCheck(cudaStreamCreate(&stream_));
  }

  TrtCudaStream(const TrtCudaStream&) = delete;

  TrtCudaStream& operator=(const TrtCudaStream&) = delete;

  TrtCudaStream(TrtCudaStream&&) = delete;

  TrtCudaStream& operator=(TrtCudaStream&&) = delete;

  ~TrtCudaStream() {
    CudaCheck(cudaStreamDestroy(stream_));
  }

  cudaStream_t Get() const {
    return stream_;
  }

  void Wait(TrtCudaEvent& event);

  void sleep(int* ms) {
#if CUDA_VERSION < 10000
    CudaCheck(cudaStreamAddCallback(stream_, CudaSleep, ms, 0));
#else
    CudaCheck(cudaLaunchHostFunc(stream_, CudaSleep, ms));
#endif
  }

 private:

  cudaStream_t stream_{};
};

//!
//! \class TrtCudaEvent
//! \brief Managed CUDA event
//!
class TrtCudaEvent {
 public:

  explicit TrtCudaEvent(bool blocking = true) {
    const unsigned int flags = blocking ? cudaEventBlockingSync : cudaEventDefault;
    CudaCheck(cudaEventCreateWithFlags(&event_, flags));
  }

  TrtCudaEvent(const TrtCudaEvent&) = delete;

  TrtCudaEvent& operator=(const TrtCudaEvent&) = delete;

  TrtCudaEvent(TrtCudaEvent&&) = delete;

  TrtCudaEvent& operator=(TrtCudaEvent&&) = delete;

  ~TrtCudaEvent() {
    CudaCheck(cudaEventDestroy(event_));
  }

  cudaEvent_t Get() const {
    return event_;
  }

  void Record(const TrtCudaStream& stream) {
    CudaCheck(cudaEventRecord(event_, stream.Get()));
  }

  void Synchronize() {
    CudaCheck(cudaEventSynchronize(event_));
  }

  // Returns time elapsed time in milliseconds
  float operator-(const TrtCudaEvent& e) const {
    float time{0};
    CudaCheck(cudaEventElapsedTime(&time, e.Get(), Get()));
    return time;
  }

 private:

  cudaEvent_t event_{};
};

inline void TrtCudaStream::Wait(TrtCudaEvent& event) {
  CudaCheck(cudaStreamWaitEvent(stream_, event.Get(), 0));
}

//!
//! \class TrtCudaBuffer
//! \brief Managed buffer for host and device
//!
template<typename A, typename D>
class TrtCudaBuffer {
 public:

  TrtCudaBuffer() = default;

  TrtCudaBuffer(const TrtCudaBuffer&) = delete;

  TrtCudaBuffer& operator=(const TrtCudaBuffer&) = delete;

  TrtCudaBuffer(TrtCudaBuffer&& rhs) {
    Reset(rhs.ptr_);
    rhs.ptr_ = nullptr;
  }

  TrtCudaBuffer& operator=(TrtCudaBuffer&& rhs) {
    if (this != &rhs) {
      Reset(rhs.ptr_);
      rhs.ptr_ = nullptr;
    }
    return *this;
  }

  ~TrtCudaBuffer() {
    Reset();
  }

  TrtCudaBuffer(size_t size) {
    A()(&ptr_, size);
  }

  void Allocate(size_t size) {
    Reset();
    A()(&ptr_, size);
  }

  void Reset(void* ptr = nullptr) {
    if (ptr_) {
      D()(ptr_);
    }
    ptr_ = ptr;
  }

  void* Get() const {
    return ptr_;
  }

 private:

  void* ptr_{nullptr};
};

struct DeviceAllocator {
  void operator()(void** ptr, size_t size) { CudaCheck(cudaMalloc(ptr, size)); }
};

struct DeviceDeallocator {
  void operator()(void* ptr) { CudaCheck(cudaFree(ptr)); }
};

struct HostAllocator {
  void operator()(void** ptr, size_t size) { CudaCheck(cudaMallocHost(ptr, size)); }
};

struct HostDeallocator {
  void operator()(void* ptr) { CudaCheck(cudaFreeHost(ptr)); }
};

using TrtDeviceBuffer = TrtCudaBuffer<DeviceAllocator, DeviceDeallocator>;

using TrtHostBuffer = TrtCudaBuffer<HostAllocator, HostDeallocator>;

//!
//! \class MirroredBuffer
//! \brief Coupled host and device buffers
//!
class MirroredBuffer {
 public:

  MirroredBuffer() = default;

  MirroredBuffer(const MirroredBuffer&) = delete;

  MirroredBuffer& operator=(const MirroredBuffer&) = delete;

  MirroredBuffer(MirroredBuffer&& other) noexcept: size_(other.size_), host_buffer_(std::move(other.host_buffer_)),
                                                   device_buffer_(std::move(other.device_buffer_)) {
  }

  MirroredBuffer& operator=(MirroredBuffer&& other) noexcept {
    if (this != &other) {
      size_ = other.size_;
      host_buffer_ = std::move(other.host_buffer_);
      device_buffer_ = std::move(other.device_buffer_);
    }
    return *this;
  }

  void Allocate(size_t size) {
    size_ = size;
    host_buffer_.Allocate(size);
    device_buffer_.Allocate(size);
  }

  void* GetDeviceBuffer() const { return device_buffer_.Get(); }

  void* GetHostBuffer() const { return host_buffer_.Get(); }

  void HostToDevice(TrtCudaStream& stream) {
    CudaCheck(cudaMemcpyAsync(device_buffer_.Get(), host_buffer_.Get(), size_, cudaMemcpyHostToDevice, stream.Get()));
  }

  void DeviceToHost(TrtCudaStream& stream) {
    CudaCheck(cudaMemcpyAsync(host_buffer_.Get(), device_buffer_.Get(), size_, cudaMemcpyDeviceToHost, stream.Get()));
  }

  int GetSize() const {
    return size_;
  }

 private:

  int size_{0};
  TrtHostBuffer host_buffer_;
  TrtDeviceBuffer device_buffer_;
};

//!
//! \class BindingInfo
//! \brief The info of trt binding.
//!
class TrtBinding {
 public:
  /**
   * Constructor.
   * @param name: Binding name.
   * @param dims: Binding dimensions shape.
   * @param vec_dim: Dimension index that the buffer is vectorized.
   * @param comps: The number of components included in one element.
   * @param is_input: If is input binding.
   * @param data_type: Data type.
   */
  explicit TrtBinding(const char* name, const nvinfer1::Dims& dims, int vec_dim, int comps, bool is_input,
                      const nvinfer1::DataType& data_type) : name_(name), dims_(dims), is_input_(is_input),
                                                             data_type_(data_type) {
    switch (data_type_) {
      case nvinfer1::DataType::kBOOL:
      case nvinfer1::DataType::kINT8:data_type_size_ = 1;
        break;
      case nvinfer1::DataType::kHALF:data_type_size_ = 2;
        break;
      case nvinfer1::DataType::kINT32:
      case nvinfer1::DataType::kFLOAT:data_type_size_ = 4;
        break;
    }

    volume_ = Volume(dims, vec_dim, comps);
    buffer_.Allocate(volume_ * data_type_size_);
  }

  MirroredBuffer& buffer() {
    return buffer_;
  }

  bool is_input() const {
    return is_input_;
  }

  uint8_t data_type_size() const {
    return data_type_size_;
  }

  int volume() const {
    return volume_;
  }

 private:
  std::string name_; //! Binding name.
  nvinfer1::Dims dims_; //! Binding dimensions shape.
  bool is_input_; //! If input binding.
  nvinfer1::DataType data_type_; //! Binding data type.
  uint8_t data_type_size_; //! Size of this data type(bytes).
  int volume_; //! Data volume.
  MirroredBuffer buffer_; //! Binding buffer.
};

#endif // TRT_TEST__TRT_CUDA_H
