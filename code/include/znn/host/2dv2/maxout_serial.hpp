#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/2dv2/host_layer.hpp"
#include "znn/host/common/pool2d/mfp.hpp"

#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace host { namespace twod {

class maxout2d_serial
    : public maxout_layer2d<host_layer2d>
{
public:
    maxout2d_serial( long_t n, long_t c, long_t fac,
                     vec2i const & is )
        : maxout_layer2d<host_layer2d>(n,c,fac,is)
    { }

private:
    void single_output( real * a, real * r ) const
    {
        for ( long_t i = 0; i < out_image_len; ++i )
        {
            *(r+i) = std::max(*(a+i), *(a+out_image_len+i));
        }

        a += out_image_len * 2;

        for ( long_t z = 2; z < factor; ++z )
        {
            for ( long_t i = 0; i < out_image_len; ++i )
            {
                *(r+i) = std::max(*(r+i), *(a+i));
            }
            a += out_image_len;
        }
    }

public:
    void forward( float* in, float* out, void* ) const override
    {
        for ( long_t i = 0; i < in_batch_size; ++i )
            for ( long_t j = 0; j < num_outputs; ++j )
            {
                this->single_output(
                    in + i * input_len + in_image_len * factor * j,
                    out + i * output_len + out_image_len * j );
            }
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
