#pragma once

#include <cudnn.h>
#include <cublas_v2.h>
#include "../handle.hpp"
#include "../utils.hpp"
#include "../memory.hpp"
#include "../device_layer.hpp"
#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../layer.hpp"
#include "cufft/utils.hpp"
#include "cufft/padded_pruned_transformer.hpp"
#include "padded_cufft.hpp"


#include <vector>

namespace znn { namespace fwd { namespace gpu {

class padded_pruned_cufft_convolutional_layer
    : public convolutional_layer_base
    , public device_layer
{
private:
    device_array<float> kernels  ;
    device_array<float> biases   ;

    cudnnTensorDescriptor_t   out_desc, bias_desc;

    std::unique_ptr<cufft_padded_pruned_forward_transformer>  input_transformer ;
    std::unique_ptr<cufft_padded_pruned_forward_transformer>  kernel_transformer;
    std::unique_ptr<cufft_padded_pruned_backward_transformer> output_transformer;

    //size_t workspace_size_ = 0;

    vec3i as;
    vec3i cs;

    float * ones;

private:
    void gemv( int m, int n, float alpha,
               const float *A, const float *x,
               float beta, float *y ) const
    {
        checkCublasErrors( cublasSgemv(handle.cublas_handle, CUBLAS_OP_N, m, n,
                                       &alpha, A, m, x, 1, &beta, y, 1) );
    }

public:
    device_array<float> forward( device_array<float> in ) const override
    {
        long_t transform_elements = cs[0] * cs[1] * cs[2];

        // Transform all the inputs
        //auto in_t = get_device_array<cuComplex>
        //(transform_elements * batch_size * num_inputs);

        std::vector<device_array<cuComplex>> in_t(batch_size);

        {
            auto tmp = get_device_array<cuComplex>
                (transform_elements * num_inputs);

            for ( long_t i = 0; i < batch_size; ++i )
            {
                in_t[i] = get_device_array<cuComplex>
                    (transform_elements * num_inputs);

                input_transformer->forward(in.get() + i * input_len,
                                           in_t[i].get(), tmp.get());
            }
        }

        in.reset();

        // We will store all the transforms here
        //auto out_t = get_device_array<cuComplex>
        //(transform_elements * batch_size * num_outputs);

        std::vector<device_array<cuComplex>> out_t(batch_size);

        for ( long_t i = 0; i < batch_size; ++i )
        {
            out_t[i] = get_device_array<cuComplex>
                (transform_elements * num_outputs);
        }

        {
            auto scratch1 = get_device_array<cuComplex>
                (transform_elements * num_inputs);
            auto scratch2 = get_device_array<cuComplex>
                (transform_elements * num_inputs);

            for ( long_t i = 0; i < num_outputs; ++i )
            {
                // fft of the kernels
                kernel_transformer->forward
                    (kernels.get() + i * num_inputs * kernel_len,
                     scratch2.get(),
                     scratch1.get());

                // Use the FFT(kernels for all the batches)
                for ( long_t j = 0; j < batch_size; ++j )
                {
                    long_t inumel = transform_elements * num_inputs ;
                    long_t onumel = transform_elements * num_outputs;

                    mul_add( scratch2.get(), scratch2.get() + inumel,
                             in_t[j].get(), scratch1.get() );

                    gemv(transform_elements * 2, num_inputs,
                         1, reinterpret_cast<float*>(scratch1.get()),
                         ones,
                         0, reinterpret_cast<float*>
                         (out_t[j].get() + i*transform_elements));

                }
            }
        }

        in_t.clear();

        auto out = get_device_array<float>(total_output_len);

        {
            auto tmp = get_device_array<cuComplex>
                (transform_elements * num_outputs);

            for ( long_t i = 0; i < batch_size; ++i )
            {
                output_transformer->backward(out_t[i].get(),
                                             out.get() + i * output_len,
                                             tmp.get());
            }
        }

        out_t.clear();

        float alpha = 1; float beta = 1;

        checkCUDNN(
            cudnnAddTensor( handle.cudnn_handle,
                            &alpha,
                            bias_desc, biases.get(),
                            &beta,
                            out_desc, out.get()) );
        // beta = 0;
        // checkCUDNN(
        //     cudnnActivationForward(
        //         handle_,
        //         CUDNN_ACTIVATION_RELU,
        //         &alpha, out_desc, out,
        //         &beta, out_desc, out) );

        return out;
    }


    ~padded_pruned_cufft_convolutional_layer()
    {
        checkCUDNN( cudnnDestroyTensorDescriptor(out_desc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(bias_desc) );
        checkCudaErrors( cudaFree( ones ));
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
    padded_pruned_cufft_convolutional_layer( long_t n, long_t fin, long_t fout,
                                             vec3i const & is, vec3i const & ks,
                                             float* km = nullptr,
                                             float* bs = nullptr )
        : convolutional_layer_base(n,fin,fout,is,ks)
        , kernels(get_device_array<float>(kernels_len))
        , biases(get_device_array<float>(fout))
    {
        {
            float* onesh = new float[fin];
            std::fill_n( onesh, fin, static_cast<float>(1));
            checkCudaErrors( cudaMalloc( &ones, fin * sizeof(float)));
            checkCudaErrors( cudaMemcpy( ones, onesh, fin * sizeof(float),
                                         cudaMemcpyHostToDevice ) );
            delete onesh;
        }

        if ( km )
        {
            device_copy_n(km, kernels_len, kernels);
        }
        if ( bs )
        {
            device_copy_n(bs, fout, biases);
        }

        vec3i os = out_image_size;

        create_tensor_descriptor(&out_desc,n,fout,os[0],os[1],os[2]);
        create_tensor_descriptor(&bias_desc,1,fout,1,1,1);

        as = details::get_optimal_size(is);

        cs = as;
        cs[2] /= 2; cs[2] += 1;


        input_transformer
            = std::unique_ptr<cufft_padded_pruned_forward_transformer>
            ( new cufft_padded_pruned_forward_transformer(is,as,fin));

        kernel_transformer
            = std::unique_ptr<cufft_padded_pruned_forward_transformer>
            ( new cufft_padded_pruned_forward_transformer(ks,as,fin));

        output_transformer
            = std::unique_ptr<cufft_padded_pruned_backward_transformer>
            ( new cufft_padded_pruned_backward_transformer
              (os, ks - vec3i::one, as, fout));

    }
};



}}} // namespace znn::fwd::gpu
