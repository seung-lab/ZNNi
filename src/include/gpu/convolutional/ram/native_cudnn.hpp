#pragma once

#include <cudnn.h>
#include "../../utils.hpp"
#include "../../handle.hpp"
#include "../../memory.hpp"
#include "../../device_layer.hpp"
#include "../../../types.hpp"
#include "../../../assert.hpp"
#include "../../../layer.hpp"

namespace znn { namespace fwd { namespace gpu {


class native_cudnn_convolutional_layer
    : public convolutional_layer_base
{
private:
    handle_t& handle_;

    cudnnTensorDescriptor_t      in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t      kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    size_t workspace_size_ = 0;

public:
    long_t workspace_size() const noexcept
    {
        return static_cast<long_t>(workspace_size_);
    }

    void forward( float* in,
                  float* out,
                  float* kernels,
                  float* biases,
                  float  beta,
                  float* workspace ) const noexcept
    {
        float alpha = 1;

        checkCUDNN(
            cudnnConvolutionForward(
                handle_.cudnn_handle,
                &alpha,
                in_desc, in,
                kernel_desc, kernels,
                conv_desc,
#if defined(ZNN_NO_PRECOMP_GEMM)
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
#else
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
#endif
                workspace, workspace_size_,
                &beta,
                out_desc, out) );

    }

    void apply_bias( float* out, float* biases ) const noexcept
    {
        float alpha = 1;
        float beta  = 1;

        checkCUDNN(
            cudnnAddTensor( handle_.cudnn_handle,
                            &alpha,
                            bias_desc, biases,
                            &beta,
                            out_desc, out) );
        beta = 0;

        // checkCUDNN(
        //     cudnnActivationForward(
        //         handle_,
        //         CUDNN_ACTIVATION_RELU,
        //         &alpha, out_desc, out,
        //         &beta, out_desc, out) );
    }


    ~native_cudnn_convolutional_layer()
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
    native_cudnn_convolutional_layer( handle_t& handle,
                                      long_t n, long_t fin, long_t fout,
                                      vec3i const & is, vec3i const & ks )
        : convolutional_layer_base(n,fin,fout,is,ks)
        , handle_(handle)
    {
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

#if !defined(ZNN_NO_PRECOMP_GEMM)
        {
            size_t what_size;
            checkCUDNN(
                cudnnGetConvolutionForwardWorkspaceSize(
                    handle.cudnn_handle,
                    in_desc,
                    kernel_desc,
                    conv_desc,
                    out_desc,
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                    &what_size));

            workspace_size_ = std::max(workspace_size_, what_size);
        }
#endif
    }
};

}}} // namespace znn::fwd::gpu
