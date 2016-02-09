#pragma once

#include <cstdlib>
#include "types.hpp"

namespace znn { namespace fwd {

namespace details {

#if defined(ZNN_MEM_ALIGN)

#else

template<typename T>
inline T * znn_malloc( long_t s )
{
    void* r = std::malloc(s * sizeof(T));
    if ( !r ) throw std::bad_alloc();
    return reinterpret_cast<T*>(r);
}

inline void znn_free( void* m )
{
    std::free(m);
}

#endif

template<typename T>
struct znn_host_deleter
{
    void operator()( T* m ) const
    {
        details::znn_free(m);
    }
};


} // namespace details

template<typename T>
using host_array = std::unique_ptr<T,details::znn_host_deleter<T>>;

template<typename T>
host_array<T> get_array( size_t elements )
{
    return host_array<T>((elements>0)?details::znn_malloc<T>(elements):nullptr);
}

}} // namespace znn::fwd
