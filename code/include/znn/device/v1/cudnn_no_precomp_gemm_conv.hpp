#pragma once

#include "znn/activation.hpp"
#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/device_layer.hpp"
#include "znn/device/v1/conv_data.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/cudnn.hpp"
#include "znn/device/common/handle.hpp"


namespace znn { namespace fwd { namespace device { namespace v1 {


class cudnn_no_precomp_gemm_conv
    : public conv_layer<device_layer>
    , public conv_data
{
private:
    cudnn::tensor_descriptor      in_desc, out_desc, bias_desc;
    cudnn::kernel_descriptor      kernel_desc;
    cudnn::convolution_descriptor conv_desc;

    activation activation_ = activation::none;

public:
    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

    device_tensor<float,5> forward( device_tensor<float,5> in ) const override
    {
        device_tensor<real,5> out(output_shape);

        float alpha = 1; float beta = 0;

        tryCUDNN(
            cudnnConvolutionForward(
                handle.cudnn_handle,
                &alpha,
                in_desc.handle(), in.data(),
                kernel_desc.handle(), kernels.data(),
                conv_desc.handle(),
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                nullptr, 0,
                &beta,
                out_desc.handle(), out.data()) );

        beta = 1;

        tryCUDNN(
            cudnnAddTensor( handle.cudnn_handle,
                            &alpha,
                            bias_desc.handle(), biases.data(),
                            &beta,
                            out_desc.handle(), out.data()) );
        if ( activation_ != activation::none )
        {
            beta = 0;

            cudnnActivationMode_t act_type;

            switch (activation_)
            {
            case activation::sigmoid:
                act_type = CUDNN_ACTIVATION_SIGMOID;
                break;
            case activation::relu:
                act_type = CUDNN_ACTIVATION_RELU;
                break;
            case activation::tanh:
                act_type = CUDNN_ACTIVATION_TANH;
                break;
            case activation::clipped_relu:
                act_type = CUDNN_ACTIVATION_CLIPPED_RELU;
                break;
            default:
                DIE("unknown activation");
            }

            tryCUDNN(
                cudnnActivationForward(
                    handle.cudnn_handle,
                    act_type,
                    &alpha, out_desc.handle(), out.data(),
                    &beta, out_desc.handle(), out.data()) );
        }

        return out;
    }

public:
    cudnn_no_precomp_gemm_conv( long_t n, long_t fin, long_t fout,
                                vec3i const & is, vec3i const & ks,
                                float * km = nullptr, float* bs = nullptr,
                                activation act = activation::none  )
        : conv_layer<device_layer>(n,fin,fout,is,ks)
        , conv_data(fin,fout,ks,km,bs)
        , activation_(act)
    {
        vec3i os = out_image_size;

        in_desc.set(n,fin,is[0],is[1],is[2]);
        out_desc.set(n,fout,os[0],os[1],os[2]);
        bias_desc.set(1,fout,1,1,1);

        kernel_desc.set(fout,fin,ks[0],ks[1],ks[2]);
        conv_desc.set();
    }
};



}}}} // namespace znn::fwd::device::v1
