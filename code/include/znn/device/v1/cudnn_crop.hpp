#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/device_layer.hpp"
#include "znn/device/v1/conv_data.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"
#include "znn/device/common/cudnn.hpp"

namespace znn { namespace fwd { namespace device { namespace v1 {

class cudnn_crop
    : public crop_layer<device_layer>
{
private:
    cudnn::tensor_descriptor  in_desc_     ;
    cudnn::tensor_descriptor  out_desc_    ;

    long_t off_;

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

        tryCUDNN( cudnnAddTensor(
                      handle.cudnn_handle,
                      &alpha, in_desc_.handle(), in.data() + off_,
                      &beta, out_desc_.handle(), out.data()) );

        return out;
    }

    cudnn_crop( long_t n, long_t fin,
                vec3i const & is, vec3i const & c )
        : crop_layer<device_layer>(n,fin,is,c)
    {
        vec3i os = is - c - c;

        in_desc_.set(n,fin,os[0],os[1],os[2],
                     fin*is[2]*is[1]*is[0],
                     is[2]*is[1]*is[0],
                     is[2]*is[1],
                     is[2],
                     1);

        off_ = c[2] + c[1]*is[2] + c[0]*is[2]*is[1];

        out_desc_.set(n,fin,os[0],os[1],os[2]);
    }
};


}}}} // namespace znn::fwd::device::v1
