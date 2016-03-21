#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/fusion/device/device_layer.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"
#include "znn/device/common/cudnn.hpp"
#include "znn/device/common/fft/transformer.hpp"

namespace znn { namespace fwd { namespace device { namespace fusion {

class fft_conv
    : public conv_layer<device_layer>
{
private:
    std::shared_ptr<device_tensor<float,5>> kernels;
    std::shared_ptr<device_array<float>>    biases ;

    vec3i as;
    vec3i cs;

    cufft_padded_pruned_forward_transformer  input_transformer ;
    cufft_padded_pruned_forward_transformer  kernel_transformer;
    cufft_padded_pruned_backward_transformer output_transformer;

    device_array<float> ones;

    size_t workspace_size_ = 0;

    cudnn::tensor_descriptor out_desc, bias_desc;

private:
    void gemv( int m, int n, float alpha,
               const float *A, const float *x,
               float beta, float *y ) const
    {
        checkCublasErrors( cublasSgemv(handle.cublas_handle, CUBLAS_OP_N, m, n,
                                       &alpha, A, m, x, 1, &beta, y, 1) );
    }

public:
    device_tensor<float,5> forward( device_tensor<float,5> in ) const override
    {
        device_array<char> ws(workspace_size_);

        long_t transform_elements = cs[0] * cs[1] * cs[2];

        std::vector<device_array<cuComplex>> in_t(batch_size);

        {
            device_array<cuComplex> tmp(transform_elements * num_inputs);

            for ( long_t i = 0; i < batch_size; ++i )
            {
                in_t[i] = device_array<cuComplex>
                    (transform_elements * num_inputs);

                input_transformer.forward(in.data() + i * input_len,
                                          in_t[i].data(),
                                          tmp.data(),
                                          ws.data());
            }
        }

        in.reset();

        std::vector<device_array<cuComplex>> out_t(batch_size);

        for ( long_t i = 0; i < batch_size; ++i )
        {
            out_t[i] = device_array<cuComplex>
                (transform_elements * num_outputs);
        }

        {
            device_array<cuComplex> scratch1(transform_elements * num_inputs);
            device_array<cuComplex> scratch2(transform_elements * num_inputs);

            for ( long_t i = 0; i < num_outputs; ++i )
            {
                // fft of the kernels
                kernel_transformer.forward
                    (kernels->data() + i * num_inputs * kernel_len,
                     scratch2.data(),
                     scratch1.data(),
                     ws.data());

                // Use the FFT(kernels for all the batches)
                for ( long_t j = 0; j < batch_size; ++j )
                {
                    long_t inumel = transform_elements * num_inputs ;

                    mul_add( scratch2.data(), scratch2.data() + inumel,
                             in_t[j].data(), scratch1.data() );

                    gemv(transform_elements * 2, num_inputs,
                         1, reinterpret_cast<float*>(scratch1.data()),
                         ones.data(),
                         0, reinterpret_cast<float*>
                         (out_t[j].data() + i*transform_elements));

                }
            }
        }

        in_t.clear();

        device_tensor<float,5> out(output_shape);

        {
            device_array<cuComplex> tmp(transform_elements * num_outputs);

            for ( long_t i = 0; i < batch_size; ++i )
            {
                output_transformer.backward(out_t[i].data(),
                                            out.data() + i * output_len,
                                            tmp.data(), ws.data());
            }
        }

        out_t.clear();

        float alpha = 1; float beta = 1;

        checkCUDNN(
            cudnnAddTensor( handle.cudnn_handle,
                            &alpha,
                            bias_desc.handle(), biases->data(),
                            &beta,
                            out_desc.handle(), out.data()) );
        // beta = 0;
        // checkCUDNN(
        //     cudnnActivationForward(
        //         handle_,
        //         CUDNN_ACTIVATION_RELU,
        //         &alpha, out_desc, out,
        //         &beta, out_desc, out) );

        return out;
    }


public:
    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory;
    }

    long_t working_memory() const override
    {
        long_t transform_elements = cs[0] * cs[1] * cs[2] * 2; // in floats

        long_t usage1
            = total_input_len
            + (batch_size + 1) * num_inputs * transform_elements;

        long_t usage2
            = (batch_size + 2) * num_inputs * transform_elements
            + batch_size * num_outputs * transform_elements;

        long_t usage3
            = (batch_size + 1) * num_outputs * transform_elements
            + total_output_len;

        return std::max({usage1,usage2,usage3}) * sizeof(float)
            + workspace_size_;
    }

    char const * name() const override
    {
        return "fft_conv";
    }

    fft_conv( long_t n, long_t fin, long_t fout,
              vec3i const & is, vec3i const & ks,
              std::shared_ptr<device_tensor<float,5>> const & kd,
              std::shared_ptr<device_array<float>> const & bd )
        : conv_layer<device_layer>(n,fin,fout,is,ks)
        , kernels(kd)
        , biases(bd)
        , as(detail::get_optimal_size(is))
        , cs(as[0],as[1],as[2]/2+1)
        , input_transformer(is,as,fin)
        , kernel_transformer(ks,as,fin)
        , output_transformer(out_image_size,ks-vec3i::one,as,fout)
        , ones(fin)
    {
        {
            host_array<float> onesh(fin);
            std::fill_n( onesh.data(), fin, static_cast<float>(1) );
            ones = onesh;
        }

        vec3i os = out_image_size;

        out_desc.set(n,fout,os[0],os[1],os[2]);
        bias_desc.set(1,fout,1,1,1);

        workspace_size_ = std::max({input_transformer.workspace_size(),
                    kernel_transformer.workspace_size(),
                    output_transformer.workspace_size()});
    }
};

}}}} // namespace znn::fwd::device::fusion
