#include <iostream>
#include <fstream>
#include <vector>

#include "cuda_runtime_api.h"
#include "common_utils.hpp"

MyLogger g_logger;

nvinfer1::ICudaEngine* loadEngine(const std::string& engine, int DLACore=-1) {
  std::ifstream engine_file(engine, std::ios::binary);
  if (!engine_file) {
    std::cout << "Error opening engine file: " << engine << std::endl;
    return nullptr;
  }

  engine_file.seekg(0, engine_file.end);
  long int fsize = engine_file.tellg();
  engine_file.seekg(0, engine_file.beg);

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

int main(int argc, char** argv) {
  cudaSetDevice(0);
  loadEngine(argv[1]);

  return 0;
}
