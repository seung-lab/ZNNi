#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/device_layer.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"
#include "znn/device/common/cudnn2d.hpp"

namespace znn { namespace fwd { namespace device { namespace v1 {

class maxout
    : public maxout_layer<device_layer>
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
            vec3i const & is )
        : maxout_layer<device_layer>(n,c,fac,is)
    {
        in_desc_.set(n,c/fac,fac,is[0]*is[1],is[2]);
        out_desc_.set(n,c/fac,1,is[0]*is[1],is[2]);
        pooling_desc_.set(fac,1,1);
    }
};


}}}} // namespace znn::fwd::device::twod
