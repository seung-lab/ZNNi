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


class cudnn_single_output_convolutional_layer
    : public convolutional_layer_base
    , public device_layer
{
private:
    handle_t& handle_;

    device_array<float> kernels  ;
    device_array<float> biases   ;

    cudnnTensorDescriptor_t      in_desc, out_desc, outb_desc, bias_desc;
    cudnnFilterDescriptor_t      kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    size_t workspace_size_ = 0;


public:
    device_array<float> forward( device_array<float> in ) const override
    {
        auto out = get_device_array<float>(total_output_len);

        void * workspace = NULL;

        if ( workspace_size_ )
        {
            checkCudaErrors( cudaMalloc(&workspace, workspace_size_ ));
        }

        for ( long_t i = 0; i < batch_size; ++i )
        {

            float alpha = 1; float beta = 0;

            for ( long_t j = 0; j < num_outputs; ++j )
            {

                checkCUDNN(
                    cudnnConvolutionForward(
                        handle_.cudnn_handle,
                        &alpha,
                        in_desc, in.get() + i * input_len,
                        kernel_desc, kernels.get() + j * kernel_len * num_inputs,
                        conv_desc,
#if defined(ZNN_NO_PRECOMP_GEMM)
                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
#else
                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
#endif
                        workspace, workspace_size_,
                        &beta,
                        out_desc, out.get() + i * output_len + j * out_image_len) );
            }

            beta = 1;

            checkCUDNN(
                cudnnAddTensor( handle_.cudnn_handle,
                                &alpha,
                                bias_desc, biases.get(),
                                &beta,
                                outb_desc, out.get() + i * output_len) );
        }


        if ( workspace_size_ )
        {
            checkCudaErrors( cudaFree(workspace) );
        }

        //beta = 0;

        // checkCUDNN(
        //     cudnnActivationForward(
        //         handle_,
        //         CUDNN_ACTIVATION_RELU,
        //         &alpha, out_desc, out,
        //         &beta, out_desc, out) );

        return out;
    }


    ~cudnn_single_output_convolutional_layer()
    {
        checkCUDNN( cudnnDestroyTensorDescriptor(in_desc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(out_desc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(outb_desc) );

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
    cudnn_single_output_convolutional_layer( handle_t& handle,
                                            long_t n, long_t fin, long_t fout,
                                            vec3i const & is, vec3i const & ks,
                                            float* km = nullptr, float* bs = nullptr )
        : convolutional_layer_base(n,fin,fout,is,ks)
        , handle_(handle)
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

        create_tensor_descriptor(&in_desc,1,fin,is[0],is[1],is[2]);
        create_tensor_descriptor(&out_desc,1,1,os[0],os[1],os[2]);
        create_tensor_descriptor(&outb_desc,1,fout,os[0],os[1],os[2]);
        create_tensor_descriptor(&bias_desc,1,fout,1,1,1);

        checkCUDNN( cudnnCreateFilterDescriptor(&kernel_desc) );
        {
            int dims[5] = { static_cast<int>(1),
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
