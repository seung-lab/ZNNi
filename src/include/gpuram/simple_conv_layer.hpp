#pragma once

#include <cudnn.h>

#include "cuda_utils.hpp"
#include "../types.hpp"
#include "../assert.hpp"
#include "../init.hpp"


namespace znn { namespace fwd { namespace gpu3dram {

class simple_conv_layer
{
private:
    cudnnHandle_t& cudnn_handle_;

    cudnnTensorDescriptor_t      in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t      filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    long_t in_memory_    ;
    long_t out_memory_   ;
    long_t kernel_memory_;
    long_t bias_memory_  ;

    size_t workspace_size_ = 0;


public:
    void forward( float * in,
                  float * out,
                  float * kernels,
                  float   beta,
                  void  * workspace ) const
    {
        float alpha = 1;

        checkCUDNN(
            cudnnConvolutionForward(
                cudnn_handle_,
                &alpha,
                in_desc, in,
                filter_desc, kernels,
                conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                workspace, workspace_size_,
                &beta,
                out_desc, out) );
    }

    void nonlinearity( float * out,
                       float * biases ) const
    {

        float alpha = 1; float beta = 1;

        checkCUDNN(
            cudnnAddTensor( cudnn_handle_,
                            &alpha,
                            bias_desc, biases,
                            &beta,
                            out_desc, out) );
        beta = 0;

        checkCUDNN(
            cudnnActivationForward(
                cudnn_handle_,
                CUDNN_ACTIVATION_RELU,
                &alpha, out_desc, out,
                &beta, out_desc, out) );
    }

    long_t in_memory() const
    {
        return in_memory_;
    }

    long_t out_memory() const
    {
        return out_memory_;
    }

    long_t kernel_memory() const
    {
        return kernel_memory_;
    }

    long_t bias_memory() const
    {
        return bias_memory_;
    }

    long_t workspace_memory() const
    {
        return static_cast<long_t>(workspace_size_);
    }

    long_t memory() const
    {
        return in_memory() + out_memory() + kernel_memory()
            + bias_memory() + workspace_memory();
    }


    ~simple_conv_layer()
    {
        checkCUDNN( cudnnDestroyTensorDescriptor(in_desc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(out_desc) );

        checkCUDNN( cudnnDestroyTensorDescriptor(bias_desc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filter_desc) );

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
    simple_conv_layer( cudnnHandle_t& cudnn_handle,
                       long_t n, long_t fin, long_t fout,
                       vec3i const & is,
                       vec3i const & fs )
        : cudnn_handle_(cudnn_handle)
    {
        kernel_memory_ = fin * fout * fs[0] * fs[1] * fs[2] * sizeof(float);
        bias_memory_   = fout * sizeof(float);

        vec3i os = is + vec3i::one - fs;

        create_tensor_descriptor(&in_desc,n,fin,is[0],is[1],is[2]);
        create_tensor_descriptor(&out_desc,n,fout,os[0],os[1],os[2]);
        create_tensor_descriptor(&bias_desc,1,fout,1,1,1);

        checkCUDNN( cudnnCreateFilterDescriptor(&filter_desc) );
        {
            int dims[5] = {fout,fin,fs[0],fs[1],fs[2]};
            checkCUDNN(
                cudnnSetFilterNdDescriptor(filter_desc,
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

        in_memory_  = n * fin  * is[0] * is[1] * is[2] * sizeof(float);
        out_memory_ = n * fout * os[0] * os[1] * os[2] * sizeof(float);

        {
            size_t what_size;
            checkCUDNN(
                cudnnGetConvolutionForwardWorkspaceSize(
                    cudnn_handle,
                    in_desc,
                    filter_desc,
                    conv_desc,
                    out_desc,
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                    &what_size));

            workspace_size_ = std::max(workspace_size_, what_size);
        }
    }
};


}}} // namespace znn::fwd::gpu3dram
