#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/host_layer.hpp"
#include "znn/host/common/pool/mfp.hpp"

#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace host { namespace v1 {

class maxfilter
    : public maxfilter_layer<host_layer>
{
public:
    maxfilter( long_t n, long_t finout,
               vec3i const & is, vec3i const & ws ) noexcept
        : maxfilter_layer<host_layer>(n,finout,is,ws)
    { }

private:
    void single_image_pool( real * im, real * out ) const
    {
        vec3i istrides(in_image_size[1] * in_image_size[2], in_image_size[2], 1);

        vec3i is = in_image_size;;

        // Along z direction
        is[2] -= window_size[2] - 1;
        if ( window_size[2] == 2 ) mfp_inplace_2( im, istrides[2], is, istrides );
        if ( window_size[2] == 3 ) mfp_inplace_3( im, istrides[2], is, istrides );
        if ( window_size[2] == 4 ) mfp_inplace_4( im, istrides[2], is, istrides );

        // Along y direction
        is[1] -= window_size[1] - 1;
        if ( window_size[1] == 2 ) mfp_inplace_2( im, istrides[1], is, istrides );
        if ( window_size[1] == 3 ) mfp_inplace_3( im, istrides[1], is, istrides );
        if ( window_size[1] == 4 ) mfp_inplace_4( im, istrides[1], is, istrides );

        // Along x direction
        is[0] -= window_size[0] - 1;
        if ( window_size[0] == 2 ) mfp_inplace_2( im, istrides[0], is, istrides );
        if ( window_size[0] == 3 ) mfp_inplace_3( im, istrides[0], is, istrides );
        if ( window_size[0] == 4 ) mfp_inplace_4( im, istrides[0], is, istrides );

        vec3i ostrides( out_image_size[2] * out_image_size[1], out_image_size[2], 1 );

        for ( long_t x = 0; x < out_image_size[0]; ++x )
            for ( long_t y = 0; y < out_image_size[1]; ++y )
            {
                std::copy_n( im + y * istrides[1] + x * istrides[0],
                             out_image_size[2],
                             out + y * ostrides[1] + x * ostrides[0]);
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
                            ret.data() + i * output_len + out_image_len * j);
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
