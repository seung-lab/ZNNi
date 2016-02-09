#pragma once

#include "utils.hpp"

#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../memory.hpp"
#include "../../layer.hpp"
#include "../utils/task_package.hpp"

namespace znn { namespace fwd { namespace cpu {

class pooling_layer
    : public pooling_layer_base
    , public host_layer
{
private:
    task_package & handle_;

public:
    pooling_layer( task_package& handle,
                   long_t n, long_t fin,
                   vec3i const & is,
                   vec3i const & ks )
        : pooling_layer_base( n, fin, is, ks )
        , handle_(handle)
    { }

private:
    void single_image_pool( real * im, real * out, long_t delta, void* ) const
    {
        vec3i strides(in_image_size[1] * in_image_size[2], in_image_size[2], 1);

        vec3i is = in_image_size;;

        // Along z direction
        is[2] -= window_size[2] - 1;
        if ( window_size[2] == 2 ) pool_inplace_2( im, strides[2], is, strides );
        if ( window_size[2] == 3 ) pool_inplace_3( im, strides[2], is, strides );
        if ( window_size[2] == 4 ) pool_inplace_4( im, strides[2], is, strides );

        // Along y direction
        is[1] -= window_size[1] - 1;
        if ( window_size[1] == 2 ) pool_inplace_2( im, strides[1], is, strides );
        if ( window_size[1] == 3 ) pool_inplace_3( im, strides[1], is, strides );
        if ( window_size[1] == 4 ) pool_inplace_4( im, strides[1], is, strides );

        // Along x direction
        is[0] -= window_size[0] - 1;
        if ( window_size[0] == 2 ) pool_inplace_2( im, strides[0], is, strides );
        if ( window_size[0] == 3 ) pool_inplace_3( im, strides[0], is, strides );
        if ( window_size[0] == 4 ) pool_inplace_4( im, strides[0], is, strides );

        vec3i istrides = strides * window_size;
        vec3i ostrides( out_image_size[2] * out_image_size[1],
                        out_image_size[2], 1 );

        for ( long_t x = 0; x < window_size[0]; ++x )
            for ( long_t y = 0; y < window_size[1]; ++y )
                for ( long_t z = 0; z < window_size[2]; ++z )
                {
                    pooling_separation
                        ( im + x*strides[0] + y*strides[1] + z*strides[2],
                          out, istrides, ostrides, out_image_size );
                    out += delta;
                }

    }

public:
    host_array<real> forward( host_array<real> m ) const override
    {
        host_array<real> ret = get_array<real>(total_output_len);

        for ( long_t i = 0; i < in_batch_size; ++i )
            for ( long_t j = 0; j < num_inputs; ++j )
            {
                handle_.add_task( &pooling_layer::single_image_pool,
                                  this,
                                  m.get() + i * input_len + in_image_len * j,
                                  ret.get() + i * output_len * window_len
                                  + out_image_len * j,
                                  output_len );
            }

        handle_.execute();

        return ret;
    }


};

}}} // namespace znn::fwd::cpu
