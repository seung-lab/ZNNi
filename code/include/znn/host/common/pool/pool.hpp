#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"

namespace znn { namespace fwd { namespace host {

template<typename T>
inline void pool_inplace_2( T * v, long_t vs,
                            vec3i const & sz,
                            vec3i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            for ( long_t k = 0, z = y; k < sz[2]; ++k, z += st[2] )
            {
                v[z] = std::max(v[z*vs], v[z*vs+vs]);
            }
        }
    }
}


template<typename T>
inline void pool_inplace_3( T * v, long_t vs,
                           vec3i const & sz,
                           vec3i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            for ( long_t k = 0, z = y; k < sz[2]; ++k, z += st[2] )
            {
                v[z] = std::max({v[z*vs], v[(z+1)*vs], v[(z+2)*vs]});
            }
        }
    }
}

template<typename T>
inline void pool_inplace_4( T * v, long_t vs,
                           vec3i const & sz,
                           vec3i const & st )
{
    for ( long_t i = 0, x = 0; i < sz[0]; ++i, x += st[0] )
    {
        for ( long_t j = 0, y = x; j < sz[1]; ++j, y += st[1] )
        {
            for ( long_t k = 0, z = y; k < sz[2]; ++k, z += st[2] )
            {
                v[z] = std::max({v[z*vs], v[(z+1)*vs],
                            v[(z+2)*vs], v[(z+3)*vs]});
            }
        }
    }
}


}}} // namespace znn::fwd::host
