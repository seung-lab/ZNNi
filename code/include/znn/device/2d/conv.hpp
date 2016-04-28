#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/2d/device_layer.hpp"
#include "znn/device/2d/conv_data.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/cudnn2d.hpp"
#include "znn/device/common/handle.hpp"


namespace znn { namespace fwd { namespace device { namespace twod {


class conv
    : public conv_layer2d<device_layer2d>
    , public conv_data2d
{
private:
    cudnn::tensor_descriptor2d      in_desc, out_desc, bias_desc;
    cudnn::kernel_descriptor2d      kernel_desc;
    cudnn::convolution_descriptor2d conv_desc;

    cudnnConvolutionFwdAlgo_t algo_;
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

    device_tensor<float,4> forward( device_tensor<float,4> in ) const override
    {
        //  I'm not sure what the problem is here, but when the input
        //  or output is larger than 1 Giga elements, the PRECOMP_GEMM
        //  seems to do something weird - writes outside of the memory
        //  given to the output tensor.  Same code works with
        //  IMPLICIT_GEMM.  Assume it's NVIDIA bug or undocumented
        //  limitation. Solution - limit the size to 1 Giga elements.

        if (   total_output_len > 1024*1024*1024
               || total_input_len > 1024*1024*1024  )
        {
            throw std::logic_error("in or out too big");
        }

        device_tensor<real,4> out(output_shape);

        device_array<char> workspace(workspace_size_);

        float alpha = 1; float beta = 0;

        tryCUDNN(
            cudnnConvolutionForward(
                handle.cudnn_handle,
                &alpha,
                in_desc.handle(), in.data(),
                kernel_desc.handle(), kernels.data(),
                conv_desc.handle(),
                algo_,
                workspace.data(), workspace_size_,
                &beta,
                out_desc.handle(), out.data()) );

        beta = 1;

        tryCUDNN(
            cudnnAddTensor( handle.cudnn_handle,
                            &alpha,
                            bias_desc.handle(), biases.data(),
                            &beta,
                            out_desc.handle(), out.data()) );
        beta = 0;

        // checkCUDNN(
        //     cudnnActivationForward(
        //         handle_,
        //         CUDNN_ACTIVATION_RELU,
        //         &alpha, out_desc, out,
        //         &beta, out_desc, out) );

        return out;
    }

public:
    conv( long_t n, long_t fin, long_t fout,
          vec2i const & is, vec2i const & ks,
          float * km = nullptr, float* bs = nullptr )
        : conv_layer2d<device_layer2d>(n,fin,fout,is,ks)
        , conv_data2d(fin,fout,ks,km,bs)
    {
        vec2i os = out_image_size;

        in_desc.set(n,fin,is[0],is[1]);
        out_desc.set(n,fout,os[0],os[1]);
        bias_desc.set(1,fout,1,1);

        kernel_desc.set(fout,fin,ks[0],ks[1]);
        conv_desc.set();

        {
            cudnnConvolutionFwdAlgoPerf_t algs[10];
            int n_ret;

            tryCUDNN(
                cudnnFindConvolutionForwardAlgorithm(
                    handle.cudnn_handle,
                    in_desc.handle(),
                    kernel_desc.handle(),
                    conv_desc.handle(),
                    out_desc.handle(),
                    10, &n_ret,
                    algs));

            algo_ = algs[0].algo;
            workspace_size_ = algs[0].memory;

            std::cout << "Best algo: " << algo_ << ' ' << workspace_size_ << "\n";
        }
    }
};



}}}} // namespace znn::fwd::device::twod
