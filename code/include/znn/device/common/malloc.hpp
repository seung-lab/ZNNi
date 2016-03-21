#pragma once

#include "znn/types.hpp"

#include <cuda_runtime.h>

namespace znn { namespace fwd { namespace device {

template<typename T>
inline device_ptr<T> device_malloc( size_t len )
{
    void* p = nullptr;

    if ( len > 0 )
    {
        auto status = cudaMalloc(&p, len * sizeof(T));
        if ( status != 0 )
        {
            throw std::bad_alloc();
        }
    }

    return device_ptr<T>(static_cast<T*>(p));
}

template<typename T>
inline void device_free( device_ptr<T> ptr )
{
    auto status = cudaFree(ptr.get());
    if ( status != 0 )
    {
        throw std::logic_error(cudaGetErrorString(status));
    }
}


}}} // namespace znn::fwd::device
