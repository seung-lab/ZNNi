#pragma once

#include "znn/types.hpp"
#include "znn/layer.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/cudnn.hpp"
#include "znn/device/common/handle.hpp"

#include <memory>

namespace znn { namespace fwd { namespace device { namespace ram {

class native_cudnn_conv
    : public conv_layer<layer>
{
private:
    cudnn::tensor_descriptor      in_desc, out_desc, bias_desc;
    cudnn::kernel_descriptor      kernel_desc;
    cudnn::convolution_descriptor conv_desc;

    size_t workspace_size_ = 0;

public:
    long_t workspace_size() const
    {
        return static_cast<long_t>(workspace_size_);
    }

    char const * name() const
    {
        return "native_cudnn_conv";
    }


    void forward( device_tensor<float,5> in, float* out,
                  float const * kernels,
                  float beta, void* workspace ) const
    {
        float alpha = 1;

        tryCUDNN(
            cudnnConvolutionForward(
                handle.cudnn_handle,
                &alpha,
                in_desc.handle(), in.data(),
                kernel_desc.handle(), kernels,
                conv_desc.handle(),
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                workspace, workspace_size_,
                &beta,
                out_desc.handle(), out) );
    }

    void nonlineiaruty( float* out, float const* biases ) const
    {
        float alpha = 1; float beta = 1;

        tryCUDNN(
            cudnnAddTensor( handle.cudnn_handle,
                            &alpha,
                            bias_desc.handle(), biases,
                            &beta,
                            out_desc.handle(), out) );
        // beta = 0;
        // checkCUDNN(
        //     cudnnActivationForward(
        //         handle_,
        //         CUDNN_ACTIVATION_RELU,
        //         &alpha, out_desc, out,
        //         &beta, out_desc, out) );

    }

public:
    native_cudnn_conv( long_t n, long_t fin, long_t fout,
                       vec3i const & is, vec3i const & ks )
        : conv_layer<layer>(n,fin,fout,is,ks)
    {
        vec3i os = out_image_size;

        in_desc.set(n,fin,is[0],is[1],is[2]);
        out_desc.set(n,fout,os[0],os[1],os[2]);
        bias_desc.set(1,fout,1,1,1);

        kernel_desc.set(fout,fin,ks[0],ks[1],ks[2]);
        conv_desc.set();

        {
            size_t what_size;
            tryCUDNN(
                cudnnGetConvolutionForwardWorkspaceSize(
                    handle.cudnn_handle,
                    in_desc.handle(),
                    kernel_desc.handle(),
                    conv_desc.handle(),
                    out_desc.handle(),
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                    &what_size));

            workspace_size_ = std::max(workspace_size_, what_size);
        }
    }
};

}}}} //  namespace znn::fwd::device::ram
