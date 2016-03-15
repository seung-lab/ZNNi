#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/device_layer.hpp"
#include "znn/device/v1/conv_data.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"


namespace znn { namespace fwd { namespace device { namespace v1 {


class cudnn_conv
    : public conv_layer<device_layer>
    , public conv_data
{
private:
    cudnnTensorDescriptor_t      in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t      kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    size_t workspace_size_ = 0;

public:
    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory + workspace_size_;
    }

    device_tensor<float,5> forward( device_tensor<float,5> in ) const override
    {
        device_tensor<real,5> out(output_shape);

        device_array<char> workspace(workspace_size_);

        float alpha = 1; float beta = 0;

        checkCUDNN(
            cudnnConvolutionForward(
                handle.cudnn_handle,
                &alpha,
                in_desc, in.data(),
                kernel_desc, kernels.data(),
                conv_desc,
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                workspace.data(), workspace_size_,
                &beta,
                out_desc, out.data()) );

        beta = 1;

        checkCUDNN(
            cudnnAddTensor( handle.cudnn_handle,
                            &alpha,
                            bias_desc, biases.data(),
                            &beta,
                            out_desc, out.data()) );
        beta = 0;

        // checkCUDNN(
        //     cudnnActivationForward(
        //         handle_,
        //         CUDNN_ACTIVATION_RELU,
        //         &alpha, out_desc, out,
        //         &beta, out_desc, out) );

        return out;
    }


    ~cudnn_conv()
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
    cudnn_conv( long_t n, long_t fin, long_t fout,
                vec3i const & is, vec3i const & ks,
                float * km = nullptr, float* bs = nullptr )
        : conv_layer<device_layer>(n,fin,fout,is,ks)
        , conv_data(fin,fout,ks,km,bs)
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
    }
};



}}}} // namespace znn::fwd::device::v1
