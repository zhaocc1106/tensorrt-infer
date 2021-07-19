#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

#include "cuda_runtime_api.h"
#include "common_utils.hpp"
#include "trt_cuda.hpp"

static MyLogger g_logger; // Global logger.
static const int kDefaultDimension = 350; // Default shape of dimension.
static std::vector<TrtBinding> g_bindings; // All bindings.

/**
 * Load trt engine from serialized engine file.
 * @param engine: Serialized engine file.
 * @param DLACore: DLA(Deep Learning Accelerator) Core.
 * @return Trt engine.
 */
nvinfer1::ICudaEngine* LoadEngine(const std::string& engine, int DLACore = -1) {
  std::ifstream engine_file(engine, std::ios::binary);
  if (!engine_file) {
    std::cout << "Error opening engine file: " << engine << std::endl;
    return nullptr;
  }

  engine_file.seekg(0, std::ifstream::end);
  size_t fsize = static_cast<size_t>(engine_file.tellg());
  engine_file.seekg(0, std::ifstream::beg);

  std::vector<char> engine_data(fsize);
  engine_file.read(engine_data.data(), fsize);
  if (!engine_file) {
    std::cout << "Error loading engine file: " << engine << std::endl;
    return nullptr;
  }

  TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(g_logger)};
  std::cout << "Number of dla cores: " << runtime->getNbDLACores() << std::endl;
  if (DLACore != -1) {
    runtime->setDLACore(DLACore);
  }

  return runtime->deserializeCudaEngine(engine_data.data(), fsize, nullptr);
}

/**
 * Setup interface. Check engine bindings. Set default dimension if shape is dynamic.
 * @param engine: Trt engine.
 * @param context: Trt execution context.
 */
void SetupInterface(const nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context) {
  std::stringstream ss;
  int binding_size = engine->getNbBindings();
  std::cout << "Model bindings(Input and Output) size: " << binding_size << std::endl;

  for (int b = 0; b < binding_size; b++) {
    auto dims = context->getBindingDimensions(b);

    const bool dynamic_shape = std::any_of(dims.d, dims.d + dims.nbDims, [](int dim) { return dim == -1; })
        || engine->isShapeBinding(b);

    // The shape of binding is dynamic. Set default dimension size.
    if (dynamic_shape) {
      nvinfer1::Dims static_dims{};
      static_dims.nbDims = dims.nbDims;
      std::transform(dims.d,
                     dims.d + dims.nbDims,
                     static_dims.d,
                     [&](int dim) { return dim > 0 ? dim : kDefaultDimension; });
      context->setBindingDimensions(b, static_dims);
    }

    dims = context->getBindingDimensions(b);
    const auto vec_dim = engine->getBindingVectorizedDim(b);
    const auto comps = engine->getBindingComponentsPerElement(b);
    const auto data_type = engine->getBindingDataType(b);
    const auto name = engine->getBindingName(b);
    const auto is_input = engine->bindingIsInput(b);

    g_bindings.emplace_back(TrtBinding(name, dims, vec_dim, comps, is_input, data_type));

    ss.str("");
    for (int d = 0; d < dims.nbDims; d++) {
      ss << dims.d[d];
      if (d < dims.nbDims - 1) {
        ss << "x";
      }
    }

    std::cout << "Binding name: " << name << ", data_type: " << (int) data_type << ", is_input: " << is_input
              << ", vec_dim: " << vec_dim << ", comps: " << comps << ", shape: " << ss.str() << std::endl;
  }
}

/**
 * Fill random input(uniform distribution) into input buffer.
 */
void FillRandomInput() {
  std::cout << "FillRandomInput" << std::endl;
  for (auto& bind : g_bindings) {
    if (bind.is_input()) { // If is input binding buffer.
      FillBuffer(bind.buffer().GetHostBuffer(), bind.volume(), (float) -1.0,
                 (float) 1.0);
    }
  }
}

/**
 * Run infer.
 * @param context: Trt execution context.
 */
void RunInfer(nvinfer1::IExecutionContext* context) {
  std::cout << "<<<<<<RunInfer" << std::endl;

  void* dev_ptr[g_bindings.size()]; // Prepare device buffer.
  for (int i = 0; i < g_bindings.size(); i++) {
    dev_ptr[i] = g_bindings[i].buffer().GetDeviceBuffer();
  }

  long long start = std::chrono::system_clock::now().time_since_epoch().count();
  TrtCudaStream cuda_stream;
  TrtCudaEvent start_event;
  start_event.Record(cuda_stream);

  for (auto& bind : g_bindings) {
    if (bind.is_input()) { // If is input binding buffer.
      bind.buffer().HostToDevice(cuda_stream); // Copy input data from host to gpu device.
    }
  }

  context->enqueueV2(dev_ptr, cuda_stream.Get(), nullptr); // Engine infer.

  for (auto& bind : g_bindings) {
    if (!bind.is_input()) { // If is output binding buffer.
      bind.buffer().DeviceToHost(cuda_stream); // Copy output data from gpu device to host.
    }
  }

  TrtCudaEvent end_event;
  end_event.Record(cuda_stream);
  end_event.Synchronize();

  long long end = std::chrono::system_clock::now().time_since_epoch().count();
  std::cout << "Cuda event ElapsedTime: " << end_event - start_event << " ms." << std::endl;
  std::cout << "Host time used " << (double) (end - start) / 1e6 << " ms." << std::endl;

  for (auto& bind : g_bindings) {
    if (!bind.is_input()) { // Show output.
      auto* ptr = (float*) bind.buffer().GetHostBuffer();
      std::cout << "<<<<<<Output:" << std::endl;
      for (int i = 0; i < bind.volume(); i++) {
        std::cout << ptr[i] << ", ";
      }
      std::cout << std::endl;
    }
  }
}

int main(int argc, char** argv) {
  cudaSetDevice(0);
  nvinfer1::ICudaEngine* engine = LoadEngine(argv[1]); // Load trt engine.
  if (engine) {
    std::cout << "engine load successfully!" << std::endl;
  } else {
    std::cout << "engine load failed!" << std::endl;
    return -1;
  }

  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  SetupInterface(engine, context); // Setup trt binding.
  FillRandomInput(); // Fill input with random.
  for (int i = 0; i < 500; i++) {
    RunInfer(context); // Trt run infer.
  }

  g_bindings.clear(); // Cuda mem operation need done before gpu device reset.
  CudaCheck(cudaDeviceReset());
  return 0;
}
