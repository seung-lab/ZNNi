#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/device_layer.hpp"
#include "znn/device/v1/conv_data.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"
#include "znn/device/common/cudnn.hpp"

namespace znn { namespace fwd { namespace device { namespace v1 {

class cudnn_mfp
    : public mfp_layer<device_layer>
{
private:
    cudnn::tensor_descriptor  in_desc_     ;
    cudnn::tensor_descriptor  out_desc_    ;
    cudnn::pooling_descriptor pooling_desc_;

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

        long_t delta = num_inputs * out_image_len;

        float alpha = 1;
        float beta  = 0;

        vec3i is = in_image_size;

        float * out_ptr = out.data();

        for ( long_t x = 0; x < window_size[0]; ++x )
            for ( long_t y = 0; y < window_size[1]; ++y )
                for ( long_t z = 0; z < window_size[2]; ++z )
                {
                    tryCUDNN( cudnnPoolingForward(
                                  handle.cudnn_handle,
                                  pooling_desc_.handle(),
                                  &alpha, in_desc_.handle(),
                                  in.data() + x*is[1]*is[2] + y*is[2] + z,
                                  &beta, out_desc_.handle(), out_ptr) );

                    out_ptr += delta;
                }

        return out;
    }

    cudnn_mfp( long_t n, long_t c,
               vec3i const & is, vec3i const & ws )
        : mfp_layer<device_layer>(n,c,is,ws)
    {
        vec3i eis = out_image_size * ws;

        in_desc_.set(n,c,eis[0],eis[1],eis[2],
                     c*is[0]*is[1]*is[2],
                     is[0]*is[1]*is[2],
                     is[1]*is[2],
                     is[2],
                     1);

        vec3i os = out_image_size;

        out_desc_.set(n,c,os[0],os[1],os[2],
                      c*out_image_len*num_fragments,
                      os[0]*os[1]*os[2],
                      os[1]*os[2],
                      os[2],
                      1);

        pooling_desc_.set(ws[0],ws[1],ws[2]);
    }
};


}}}} // namespace znn::fwd::device::v1
