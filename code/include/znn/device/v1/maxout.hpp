#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/device_layer.hpp"
#include "znn/device/v1/conv_data.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"
#include "znn/device/common/cudnn.hpp"
#include "znn/device/common/kernels.hpp"

namespace znn { namespace fwd { namespace device { namespace v1 {

class maxout
    : public maxout_layer<device_layer>
{
public:
    device_tensor<float,5> forward( device_tensor<float,5> in ) const override
    {
        device_tensor<float,5> out(output_shape);

        float* binp  = in.data();
        float* boutp = out.data();

        for ( long_t i = 0; i < batch_size; ++i )
        {
            for ( long_t j = 1; j < factor; ++j )
            {
                max_out_transform(binp + j * output_len,
                                  binp + j * output_len + output_len,
                                  (j==1) ? binp : boutp,
                                  boutp);
            }

            binp  += input_len ;
            boutp += output_len;
        }

        return out;
    }


public:
    long_t resident_memory() const override
    {
        return 0;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

    maxout( long_t n, long_t fin, long_t fac, vec3i const & is )
        : maxout_layer<device_layer>(n,fin,fac,is)
    {}
};



}}}} // namespace znn::fwd::device::v1
