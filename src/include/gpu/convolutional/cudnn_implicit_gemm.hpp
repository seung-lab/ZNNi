#pragma once

#include <cudnn.h>
#include "../utils.hpp"
#include "../handle.hpp"
#include "../memory.hpp"
#include "../device_layer.hpp"
#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../layer.hpp"

namespace znn { namespace fwd { namespace gpu {


class cudnn_implicit_gemm_convolutional_layer
    : public convolutional_layer_base
    , public device_layer
{
private:
    device_array<float> kernels  ;
    device_array<float> biases   ;

    cudnnTensorDescriptor_t      in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t      kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;

public:
    long_t permanent_memory_required() const override
    {
        return kernel_memory;
    }

    long_t working_memory_required() const override
    {
        return input_memory + output_memory;
    }

    device_array<float> forward( device_array<float> in ) const override
    {
        auto out = get_device_array<float>(total_output_len);

        float alpha = 1; float beta = 0;

        checkCUDNN(
            cudnnConvolutionForward(
                handle.cudnn_handle,
                &alpha,
                in_desc, in.get(),
                kernel_desc, kernels.get(),
                conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                nullptr, 0,
                &beta,
                out_desc, out.get()) );

        beta = 1;

        checkCUDNN(
            cudnnAddTensor( handle.cudnn_handle,
                            &alpha,
                            bias_desc, biases.get(),
                            &beta,
                            out_desc, out.get()) );
        beta = 0;

        // checkCUDNN(
        //     cudnnActivationForward(
        //         handle_,
        //         CUDNN_ACTIVATION_RELU,
        //         &alpha, out_desc, out,
        //         &beta, out_desc, out) );

        return out;
    }


    ~cudnn_implicit_gemm_convolutional_layer()
    {
        checkCUDNN( cudnnDestroyTensorDescriptor(in_desc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(out_desc) );

        checkCUDNN( cudnnDestroyTensorDescriptor(bias_desc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(kernel_desc) );

        checkCUDNN( cudnnDestroyConvolutionDescriptor(conv_desc) );
    }

private:
    void create_tensor_descriptor( cudnnTensorDescriptor_t * descriptor,
                                   int n, int c, int d, int h, int w )
    {
        checkCUDNN( cudnnCreateTensorDescriptor(descriptor) );

        int dims[5] = {n,c,d,h,w};
        int strides[5] = {c*d*h*w,d*h*w,h*w,w,1};
        checkCUDNN(
            cudnnSetTensorNdDescriptor(*descriptor,
                                       CUDNN_DATA_FLOAT,
                                       5, dims, strides));
    }

public:
    cudnn_implicit_gemm_convolutional_layer( long_t n, long_t fin, long_t fout,
                                             vec3i const & is, vec3i const & ks,
                                             float* km = nullptr,
                                             float* bs = nullptr )
        : convolutional_layer_base(n,fin,fout,is,ks)
        , kernels(get_device_array<float>(kernels_len))
        , biases(get_device_array<float>(fout))
    {
        if ( km )
        {
            device_copy_n(km, kernels_len, kernels);
        }
        if ( bs )
        {
            device_copy_n(bs, fout, biases);
        }

        vec3i os = out_image_size;

        create_tensor_descriptor(&in_desc,n,fin,is[0],is[1],is[2]);
        create_tensor_descriptor(&out_desc,n,fout,os[0],os[1],os[2]);
        create_tensor_descriptor(&bias_desc,1,fout,1,1,1);

        checkCUDNN( cudnnCreateFilterDescriptor(&kernel_desc) );
        {
            int dims[5] = { static_cast<int>(fout),
                            static_cast<int>(fin),
                            static_cast<int>(ks[0]),
                            static_cast<int>(ks[1]),
                            static_cast<int>(ks[2]) };
            checkCUDNN(
                cudnnSetFilterNdDescriptor(kernel_desc,
                                           CUDNN_DATA_FLOAT,
                                           5, dims));
        }

        checkCUDNN( cudnnCreateConvolutionDescriptor(&conv_desc) );
        {
            int pad[3] = {0,0,0};
            int ones[3] = {1,1,1};

            checkCUDNN(
                cudnnSetConvolutionNdDescriptor(
                    conv_desc,
                    3, pad, ones, ones,
                    CUDNN_CONVOLUTION,
                    //CUDNN_CROSS_CORRELATION,
                    CUDNN_DATA_FLOAT) );

        }
    }
};



}}} // namespace znn::fwd::gpu
