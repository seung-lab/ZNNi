#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/2dv2/device_layer.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"
#include "znn/device/common/cudnn2d.hpp"

namespace znn { namespace fwd { namespace device { namespace twod {

class softmax
    : public softmax_layer2d<device_layer2d>
{
private:
    cudnn::tensor_descriptor2d  in_desc_     ;
    cudnn::tensor_descriptor2d  out_desc_    ;

public:
    long_t resident_memory() const override
    {
        return 0;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

    void forward( float* in, float* out, void* ) const override
    {
        float alpha = 1;
        float beta  = 0;

        tryCUDNN( cudnnSoftmaxForward(
                      handle.cudnn_handle,
                      CUDNN_SOFTMAX_FAST,
                      CUDNN_SOFTMAX_MODE_CHANNEL,
                      &alpha, in_desc_.handle(), in,
                      &beta, out_desc_.handle(), out) );

    }

    softmax( long_t n, long_t c, vec2i const & is )
        : softmax_layer2d<device_layer2d>(n,c,is)
    {
        in_desc_.set(n,c,is[0],is[1]);
        out_desc_.set(n,c,is[0],is[1]);
    }
};


}}}} // namespace znn::fwd::device::twod
