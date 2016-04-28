#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/2d/device_layer.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"
#include "znn/device/common/cudnn2d.hpp"

namespace znn { namespace fwd { namespace device { namespace twod {

class maxout
    : public maxout_layer2d<device_layer2d>
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
        device_tensor<real,4> out(output_shape);

        float alpha = 1;
        float beta  = 0;

        tryCUDNN( cudnnPoolingForward(
                      handle.cudnn_handle,
                      pooling_desc_.handle(),
                      &alpha, in_desc_.handle(), in.data(),
                      &beta, out_desc_.handle(), out.data()) );

        return out;
    }

    maxout( long_t n, long_t c, long_t fac,
            vec2i const & is )
        : maxout_layer2d<device_layer2d>(n,c,fac,is)
    {
        in_desc_.set(n,c/fac,fac,is[0]*is[1]);
        out_desc_.set(n,c/fac,1,is[0]*is[1]);
        pooling_desc_.set(fac,1);
    }
};


}}}} // namespace znn::fwd::device::twod
