#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/2d/device_layer.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"
#include "znn/device/common/cudnn2d.hpp"

namespace znn { namespace fwd { namespace device { namespace twod {

class mfp
    : public mfp_layer2d<device_layer2d>
{
private:
    cudnn::tensor_descriptor2d  in_desc_     ;
    cudnn::tensor_descriptor2d  out_desc_    ;
    cudnn::pooling_descriptor2d pooling_desc_;

public:
    long_t resident_memory() const override
    {
        return 0;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

    device_tensor<float,4> forward( device_tensor<float,4> in ) const override
    {
	//std::cout << "MFP: " << output_shape << "\n";
        device_tensor<real,4> out(output_shape);

	//return out;

        long_t delta = num_inputs * out_image_len;

        float alpha = 1;
        float beta  = 0;

        vec2i is = in_image_size;

        float * out_ptr = out.data();

        for ( long_t x = 0; x < window_size[0]; ++x )
            for ( long_t y = 0; y < window_size[1]; ++y )
            {
                tryCUDNN( cudnnPoolingForward(
                              handle.cudnn_handle,
                              pooling_desc_.handle(),
                              &alpha, in_desc_.handle(),
                              in.data() + x*is[1] + y,
                              &beta, out_desc_.handle(), out_ptr) );

                out_ptr += delta;
            }

        return out;
    }

    mfp( long_t n, long_t c,
         vec2i const & is, vec2i const & ws )
        : mfp_layer2d<device_layer2d>(n,c,is,ws)
    {
        vec2i eis = out_image_size * ws;

        in_desc_.set(n,c,eis[0],eis[1],
                     c*is[0]*is[1],
                     is[0]*is[1],
                     is[1],
                     1);

        vec2i os = out_image_size;

        out_desc_.set(n,c,os[0],os[1],
                      c*out_image_len*num_fragments,
                      os[0]*os[1],
                      os[1],
                      1);

        pooling_desc_.set(ws[0],ws[1]);
    }
};


}}}} // namespace znn::fwd::device::twod
