#pragma once

#include "znn/activation.hpp"
#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/device_layer.hpp"
#include "znn/device/v1/bias_data.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/cudnn.hpp"
#include "znn/device/common/handle.hpp"


namespace znn { namespace fwd { namespace device { namespace v1 {


class cudnn_activation
    : public activation_layer<device_layer>
    , public bias_data
{
private:
    cudnn::tensor_descriptor      inout_desc, bias_desc;

    activation activation_ = activation::none;
    cudnnDataType_t dtype_;
    cudnnActivationMode_t mode_;
    cudnnTensorDescriptor_t shape_desc_;

    #if CUDNN_MAJOR == 5
      cudnnActivationDescriptor_t act_desc_;
      cudnnNanPropagation_t nan_prop_;
      double relu_ceil_;
    #endif

    //float alpha, beta;

public:
    long_t resident_memory() const override
    {
        return bias_memory;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

    device_tensor<float,5> forward( device_tensor<float,5> in ) const override
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

        float alpha = 1;
        float beta = 1;

        tryCUDNN(
          cudnnAddTensor( handle.cudnn_handle,
                          &alpha,
                          bias_desc.handle(), biases.data(),
                          &beta,
                          inout_desc.handle(), in.data()) );

        beta = 0;
        if ( activation_ != activation::none )
        {
          cudnnActivationMode_t mode_;
          switch (activation_)
          {
          case activation::sigmoid:
              mode_ = CUDNN_ACTIVATION_SIGMOID;
              break;
          case activation::relu:
              mode_ = CUDNN_ACTIVATION_RELU;
              break;
          case activation::tanh:
              mode_ = CUDNN_ACTIVATION_TANH;
              break;
          case activation::clipped_relu:
              mode_ = CUDNN_ACTIVATION_CLIPPED_RELU;
              break;
          default:
              DIE("unknown activation");
          }

          tryCUDNN(
            cudnnActivationForward_v3(
                handle.cudnn_handle,
                mode_,
                &alpha, inout_desc.handle(), in.data(),
                &beta, inout_desc.handle(), in.data()) );
        }
        return in;
    }

public:
    cudnn_activation( long_t n, long_t finout,
                      vec3i const & is, float* bs = nullptr,
                      activation act = activation::none )
        : activation_layer<device_layer>(n,finout,is)
        , bias_data(finout,bs)
        , activation_(act)
    {
      inout_desc.set(n,finout,is[0],is[1],is[2]);
      bias_desc.set(1,finout,1,1,1);
    }
};

}}}} // namespace znn::fwd::device::v1
