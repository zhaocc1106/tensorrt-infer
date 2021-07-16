cmake_minimum_required(VERSION 3.16)
include(cmake/modules/set_ifndef.cmake)
include(cmake/modules/find_library_create_target.cmake)

project(trt_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")

set(TRT_DEBUG_POSTFIX _debug CACHE STRING "suffix for debug builds")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in debug mode ${DEBUG_POSTFIX}")
endif ()

if (DEFINED GPU_ARCHS)
    message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
    separate_arguments(GPU_ARCHS)
else ()
    list(APPEND GPU_ARCHS
            35
            53
            61
            70
            75
            )
    message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif ()

# Generate SASS for each architecture
foreach (arch ${GPU_ARCHS})
    if (${arch} GREATER_EQUAL 70)
        set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endif ()
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach ()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if (${LATEST_SM} GREATER_EQUAL 70)
    set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif ()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

################################### CUDA CUDNN TRT DEPENDENCIES ##########################################
set(DEFAULT_CUDA_VERSION 10.0)
set(DEFAULT_CUDNN_VERSION 7.6)
set(DEFAULT_TRT_VERSION 7.0.0.11)

set_ifndef(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
message(STATUS "CUDA version set to ${CUDA_VERSION}")
set_ifndef(CUDNN_VERSION ${DEFAULT_CUDNN_VERSION})
message(STATUS "cuDNN version set to ${CUDNN_VERSION}")
set_ifndef(TRT_VERSION ${DEFAULT_TRT_VERSION})
message(STATUS "cuDNN version set to ${TRT_VERSION}")
set_ifndef(CUDA_INSTALL_DIR ${CUDA_TOOLKIT_ROOT_DIR})
message(STATUS "CUDA_INSTALL_DIR: ${CUDA_INSTALL_DIR}")
set_ifndef(TENSORRT_INSTALL_DIR /usr/local/TensorRT-7.0.0.11)
message(STATUS "TENSORRT_INSTALL_DIR: ${TENSORRT_INSTALL_DIR}")

find_library(CUDNN_LIB cudnn HINTS
        ${CUDA_TOOLKIT_ROOT_DIR} ${CUDNN_ROOT_DIR} PATH_SUFFIXES lib64 lib)
find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/stubs)
find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib64)
find_library(RT_LIB rt)
find_library(TRT_LIB nvinfer HINTS ${TENSORRT_INSTALL_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib)
message(STATUS "CUDNN_LIB: ${CUDNN_LIB}")
message(STATUS "CUBLAS_LIB: ${CUBLAS_LIB}")
message(STATUS "CUDART_LIB: ${CUDART_LIB}")
message(STATUS "RT_LIB: ${RT_LIB}")
message(STATUS "TRT_LIB: ${TRT_LIB}")

set(SAMPLE_DEP_LIBS
        ${CUDART_LIB}
        ${CUBLAS_LIB}
        ${CUDNN_LIB}
        ${TRT_LIB}
        ${RT_LIB}
        )
################################### CUDA CUDNN TRT DEPENDENCIES ##########################################

add_executable(trt_test main.cpp common_utils.hpp)

target_include_directories(trt_test
        PUBLIC ${CUDA_INSTALL_DIR}/include
        PUBLIC ${TENSORRT_INSTALL_DIR}/include
        )

target_compile_options(trt_test PUBLIC "-fno-rtti")

target_link_libraries(trt_test
        ${SAMPLE_DEP_LIBS}
        -Wl,--unresolved-symbols=ignore-in-shared-libs
        )

set_target_properties(trt_test PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

set_target_properties(trt_test PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})