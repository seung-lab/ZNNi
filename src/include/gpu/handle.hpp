#pragma once

#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>

#include <zi/utility/singleton.hpp>

#include "utils.hpp"

namespace znn { namespace fwd { namespace gpu {

struct handle_t
{
    cudnnHandle_t  cudnn_handle ;
    cublasHandle_t cublas_handle;

    int cudnn_version;

    double device_memory;

    handle_t()
    {
        cudnn_version = static_cast<int>(cudnnGetVersion());
        printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h"
               " : %d (%s)\n",
               version, CUDNN_VERSION, CUDNN_VERSION_STR);

        printf("Host compiler version : %s %s\r", COMPILER_NAME,
               COMPILER_VER);
        showCudaDevices();

        int device = 0;
        checkCudaErrors( cudaSetDevice(device) );
        std::cout << "Using device " << device << std::endl;

        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties( &prop, device ));
        device_memory = prop.totalGlobalMem/double(1024*1024);

        std::cout << "Memory: " << device_memory << std::endl;


        checkCUDNN( cudnnCreate(&cudnn_handle) );
        checkCublasErrors( cublasCreate(&cublas_handle) );
    }

    ~handle_t()
    {
        checkCUDNN( cudnnDestroy(cudnn_handle) );
        checkCublasErrors( cublasDestroy(cublas_handle) );
        cudaDeviceReset();
    }

    handle_t(const handle_t&) = delete;
    handle_t& operator=(const handle_t&) = delete;

    handle_t(handle_t&& other) = delete;
    handle_t& operator=(handle_t&&) = delete;
};

namespace {
handle_t& handle =
    zi::singleton<handle_t>::instance();
}


}}} // namespace znn::fwd::cpu
