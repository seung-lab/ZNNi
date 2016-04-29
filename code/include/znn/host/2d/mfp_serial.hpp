#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/2d/host_layer.hpp"
#include "znn/host/common/pool2d/mfp.hpp"

#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace host { namespace twod {

class mfp2d_serial
    : public mfp_layer2d<host_layer2d>
{
public:
    mfp2d_serial( long_t n, long_t finout,
                  vec2i const & is, vec2i const & ws ) noexcept
        : mfp_layer2d<host_layer2d>(n,finout,is,ws)
    { }

private:
    void single_image_pool( real * im, real * out, long_t delta ) const
    {
        vec2i strides(in_image_size[1], 1);

        vec2i is = in_image_size;;

        // Along y direction
        is[1] -= window_size[1] - 1;
        if ( window_size[1] == 2 ) mfp2d_inplace_2( im, strides[1], is, strides );
        if ( window_size[1] == 3 ) mfp2d_inplace_3( im, strides[1], is, strides );
        if ( window_size[1] == 4 ) mfp2d_inplace_4( im, strides[1], is, strides );

        // Along x direction
        is[0] -= window_size[0] - 1;
        if ( window_size[0] == 2 ) mfp2d_inplace_2( im, strides[0], is, strides );
        if ( window_size[0] == 3 ) mfp2d_inplace_3( im, strides[0], is, strides );
        if ( window_size[0] == 4 ) mfp2d_inplace_4( im, strides[0], is, strides );

        vec2i istrides = strides * window_size;
        vec2i ostrides( out_image_size[1], 1 );

        for ( long_t x = 0; x < window_size[0]; ++x )
            for ( long_t y = 0; y < window_size[1]; ++y )
            {
                mfp2d_separation
                    ( im + x*strides[0] + y*strides[1],
                      out, istrides, ostrides, out_image_size );
                out += delta;
            }
    }

public:
    host_tensor<real,4> forward( host_tensor<real,4> in ) const override
    {
        host_tensor<float,4> ret(output_shape);

        for ( long_t i = 0; i < in_batch_size; ++i )
            for ( long_t j = 0; j < num_inputs; ++j )
            {
                this->single_image_pool(
                    in.data() + i * input_len + in_image_len * j,
                    ret.data() + i * output_len * num_fragments
                    + out_image_len * j,
                    output_len);
            }

        tg.wait();

        return ret;
    }

    long_t resident_memory() const override
    {
        return 0;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

};


}}}} // namespace znn::fwd::host::twod
