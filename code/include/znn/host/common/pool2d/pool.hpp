#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"

namespace znn { namespace fwd { namespace host {

template<typename T>
inline void pool2d_inplace_2( T * v, long_t vs,
                              vec2i const & sz,
                              vec2i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            v[y] = std::max(v[y*vs], v[y*vs+vs]);
        }
    }
}


template<typename T>
inline void pool2d_inplace_3( T * v, long_t vs,
                              vec2i const & sz,
                              vec2i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            v[y] = std::max({v[y*vs], v[(y+1)*vs], v[(y+2)*vs]});
        }
    }
}

template<typename T>
inline void pool2d_inplace_4( T * v, long_t vs,
                              vec2i const & sz,
                              vec2i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            v[y] = std::max({v[y*vs], v[(y+1)*vs],
                        v[(y+2)*vs], v[(y+3)*vs]});
        }
    }
}


}}} // namespace znn::fwd::host
