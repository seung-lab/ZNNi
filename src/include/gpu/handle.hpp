#pragma once

#include <cudnn.h>
#include <cublas_v2.h>
#include "utils.hpp"


namespace znn { namespace fwd { namespace gpu {

struct handle_t
{
    cudnnHandle_t  cudnn_handle ;
    cublasHandle_t cublas_handle;

    handle_t()
    {
        checkCUDNN( cudnnCreate(&cudnn_handle) );
        checkCublasErrors( cublasCreate(&cublas_handle) );
    }

    ~handle_t()
    {
        checkCUDNN( cudnnDestroy(cudnn_handle) );
        checkCublasErrors( cublasDestroy(cublas_handle) );
    }

    handle_t(const handle_t&) = delete;
    handle_t& operator=(const handle_t&) = delete;

    handle_t(handle_t&& other) = delete;
    handle_t& operator=(handle_t&&) = delete;
};

}}} // namespace znn::fwd::cpu
