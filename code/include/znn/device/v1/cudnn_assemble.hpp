#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/device_layer.hpp"
#include "znn/device/v1/conv_data.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"
#include "znn/device/common/cudnn.hpp"

namespace znn { namespace fwd { namespace device { namespace v1 {

class cudnn_assemble
    : public assemble_layer<device_layer>
{
private:
    cudnn::tensor_descriptor  in_desc_     ;
    cudnn::tensor_descriptor  out_desc_    ;

public:
    long_t resident_memory() const override
    {
        return 0;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

    device_tensor<float,5> forward( device_tensor<float,5> in ) const override
    {
        device_tensor<real,5> out(output_shape);

        long_t delta = num_inputs * in_image_len;

        float alpha = 1;
        float beta  = 0;

        vec3i os = in_image_size * window_size;

        float * in_ptr = in.data();

        for ( long_t x = 0; x < window_size[0]; ++x )
            for ( long_t y = 0; y < window_size[1]; ++y )
                for ( long_t z = 0; z < window_size[2]; ++z )
                {

                    tryCUDNN( cudnnAddTensor(
                                  handle.cudnn_handle,
                                  &alpha, in_desc_.handle(),
                                  in_ptr,
                                  &beta, out_desc_.handle(),
                                  out.data() + x*os[1]*os[2] + y*os[2] + z) );
                    in_ptr += delta;
                }

        return out;
    }

    cudnn_assemble( long_t n, long_t c,
                    vec3i const & is, vec3i const & ws )
        : assemble_layer<device_layer>(n,c,is,ws)
    {
        vec3i os = in_image_size * ws;

        in_desc_.set(n/num_fragments,c,is[0],is[1],is[2],
                     num_fragments*c*is[0]*is[1]*is[2],
                     is[0]*is[1]*is[2],
                     is[1]*is[2],
                     is[2],
                     1);

        out_desc_.set(n/num_fragments,c,is[0],is[1],is[2],
                      num_fragments*c*is[0]*is[1]*is[2],
                      os[0]*os[1]*os[2],
                      os[1]*os[2]*ws[0],
                      os[2]*ws[1],
                      ws[2]);
    }
};


}}}} // namespace znn::fwd::device::v1
