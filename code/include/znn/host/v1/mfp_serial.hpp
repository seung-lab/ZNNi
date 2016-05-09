#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/host_layer.hpp"
#include "znn/host/common/pool/mfp.hpp"

#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace host { namespace v1 {

class mfp_serial
    : public mfp_layer<host_layer>
{
public:
    mfp_serial( long_t n, long_t finout,
                vec3i const & is, vec3i const & ws ) noexcept
        : mfp_layer<host_layer>(n,finout,is,ws)
    { }

private:
    void single_image_pool( real * im, real * out, long_t delta ) const
    {
        vec3i strides(in_image_size[1] * in_image_size[2], in_image_size[2], 1);

        vec3i is = in_image_size;;

        // Along z direction
        is[2] -= window_size[2] - 1;
        if ( window_size[2] == 2 ) mfp_inplace_2( im, strides[2], is, strides );
        if ( window_size[2] == 3 ) mfp_inplace_3( im, strides[2], is, strides );
        if ( window_size[2] == 4 ) mfp_inplace_4( im, strides[2], is, strides );

        // Along y direction
        is[1] -= window_size[1] - 1;
        if ( window_size[1] == 2 ) mfp_inplace_2( im, strides[1], is, strides );
        if ( window_size[1] == 3 ) mfp_inplace_3( im, strides[1], is, strides );
        if ( window_size[1] == 4 ) mfp_inplace_4( im, strides[1], is, strides );

        // Along x direction
        is[0] -= window_size[0] - 1;
        if ( window_size[0] == 2 ) mfp_inplace_2( im, strides[0], is, strides );
        if ( window_size[0] == 3 ) mfp_inplace_3( im, strides[0], is, strides );
        if ( window_size[0] == 4 ) mfp_inplace_4( im, strides[0], is, strides );

        vec3i istrides = strides * window_size;
        vec3i ostrides( out_image_size[2] * out_image_size[1],
                        out_image_size[2], 1 );

        for ( long_t x = 0; x < window_size[0]; ++x )
            for ( long_t y = 0; y < window_size[1]; ++y )
                for ( long_t z = 0; z < window_size[2]; ++z )
                {
                    mfp_separation
                        ( im + x*strides[0] + y*strides[1] + z*strides[2],
                          out, istrides, ostrides, out_image_size );
                    out += delta;
                }

    }

public:
    host_tensor<real,5> forward( host_tensor<real,5> in ) const override
    {
        host_tensor<float,5> ret(output_shape);

        tbb::task_group tg;

        for ( long_t i = 0; i < in_batch_size; ++i )
            for ( long_t j = 0; j < num_inputs; ++j )
            {
                tg.run([&,i,j]() {
                        this->single_image_pool(
                            in.data() + i * input_len + in_image_len * j,
                            ret.data() + i * output_len * num_fragments
                            + out_image_len * j,
                            output_len);
                    });
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


}}}} // namespace znn::fwd::host::v1
