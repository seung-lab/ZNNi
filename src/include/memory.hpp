#pragma once

#include <cstdlib>
#include "types.hpp"

namespace znn { namespace fwd {

namespace details {

inline void* aligned_malloc(size_t required_bytes, size_t alignment)
{
    void* p1; // original block
    void** p2; // aligned block
    int offset = alignment - 1 + sizeof(void*);
    if ((p1 = (void*)std::malloc(required_bytes + offset)) == NULL)
    {
       return NULL;
    }
    p2 = (void**)(((size_t)(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

inline void aligned_free(void *p)
{
    std::free(((void**)p)[-1]);
}

template<typename T>
inline T * znn_malloc( long_t s )
{
void* r = aligned_malloc(s * sizeof(T), 16);
    if ( !r ) throw std::bad_alloc();
    return reinterpret_cast<T*>(r);
}

inline void znn_free( void* m )
{
    aligned_free(m);
}

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
