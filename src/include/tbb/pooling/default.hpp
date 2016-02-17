#pragma once

#include "../../cpu/pooling/utils.hpp"

#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../memory.hpp"
#include "../../layer.hpp"
#include "../host_layer.hpp"

#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace tbb {

class pooling_layer
    : public pooling_layer_base
    , public host_layer
{
public:
    pooling_layer( void*,
                   long_t n, long_t fin,
                   vec3i const & is,
                   vec3i const & ks )
        : pooling_layer_base( n, fin, is, ks )
    { }

private:
    void single_image_pool( real * im, real * out, long_t delta ) const noexcept
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

        ::tbb::task_group tg;

        for ( long_t i = 0; i < in_batch_size; ++i )
            for ( long_t j = 0; j < num_inputs; ++j )
            {
                tg.run([&m,&ret,i,j,this]() {
                        this->single_image_pool
                            ( m.get() + i * this->input_len + this->in_image_len * j,
                              ret.get() + i * this->output_len * this->window_len
                              + this->out_image_len * j,
                              this->output_len );
                            });
            }


        tg.wait();

        return ret;
    }


};

}}} // namespace znn::fwd::tbb
