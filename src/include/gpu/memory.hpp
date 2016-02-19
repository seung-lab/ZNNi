#pragma once

#include "utils.hpp"
#include "../types.hpp"
#include "../memory.hpp"

namespace znn { namespace fwd {

namespace details {

template<typename T>
struct znn_device_deleter
{
    void operator()( T* m ) const
    {
        checkCudaErrors( cudaFree(m) );
    }
};


} // namespace details

template<typename T>
using device_array = std::unique_ptr<T,details::znn_device_deleter<T>>;

template<typename T>
inline device_array<T> get_device_array( size_t elements )
{
    T* ptr = nullptr;
    if ( elements > 0 )
    {
        checkCudaErrors( cudaMalloc(&ptr, elements * sizeof(T) ));
    }

    return device_array<T>(ptr);
}

template<typename T>
inline void device_copy_n( T const * s, size_t n, device_array<T>& d)
{
    checkCudaErrors( cudaMemcpy( d.get(), s, n * sizeof(T),
                                 cudaMemcpyHostToDevice ) );
}


template<typename T>
inline void device_copy_n( host_array<T>& s, size_t n, device_array<T>& d)
{
    checkCudaErrors( cudaMemcpy( d.get(), s.get(), n * sizeof(T),
                                 cudaMemcpyHostToDevice ) );
}


template<typename T>
inline void device_copy_n( device_array<T>& s, size_t n, host_array<T>& d)
{
    checkCudaErrors( cudaMemcpy( d.get(), s.get(), n * sizeof(T),
                                 cudaMemcpyDeviceToHost ) );
}



}} // namespace znn::fwd
