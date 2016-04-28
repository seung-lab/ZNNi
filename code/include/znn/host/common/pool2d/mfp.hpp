#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"

namespace znn { namespace fwd { namespace host {

template<typename T>
inline void mfp2d_inplace_2( T * v, long_t vs,
                             vec2i const & sz,
                             vec2i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            v[y] = std::max(v[y], v[y+vs]);
        }
    }
}


template<typename T>
inline void mfp2d_inplace_3( T * v, long_t vs,
                             vec2i const & sz,
                             vec2i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            v[y] = std::max(v[y], v[y+vs]);
            v[y] = std::max(v[y], v[y+vs*2]);
        }
    }
}

template<typename T>
inline void mfp2d_inplace_4( T * v, long_t vs,
                             vec2i const & sz,
                             vec2i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            v[y] = std::max(v[y], v[y+vs]);
            v[y] = std::max(v[y], v[y+vs*2]);
            v[y] = std::max(v[y], v[y+vs*3]);
        }
    }
}


template<typename T>
inline void mfp2d_separation( T const * src,
                              T * dst,
                              vec2i const & istrides,
                              vec2i const & ostrides,
                              vec2i const & size )
{
    for ( long_t i = 0; i < size[0]; ++i )
    {
        for ( long_t j = 0; j < size[1]; ++j )
        {
            dst[i * ostrides[0] + j * ostrides[1]]
                = src[i * istrides[0] + j * istrides[1]];
        }
    }
}


}}} // namespace znn::fwd::host
