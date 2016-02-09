#pragma once

#include <cudnn.h>
#include <cublas_v2.h>
#include "../utils.hpp"
#include "../memory.hpp"
#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../layer.hpp"
#include "cufft/utils.hpp"

namespace znn { namespace fwd { namespace gpu {


// template<typename T>
// inline void examine(T* p, size_t len)
// {
//     T x[len];

//     cudaMemcpy( x, p, len * sizeof(T), cudaMemcpyDeviceToHost);

//     for ( size_t i  = 0; i < len; ++i )
//         std::cout << x[i] << ' ';

//     std::cout << "\n\n";
// }

class cufft_convolutional_layer: public convolutional_layer_base
{
private:
    cudnnHandle_t& handle_;
    cublasHandle_t cublas_handle_;

    device_array<float> kernels  ;
    device_array<float> biases   ;

    cudnnTensorDescriptor_t   out_desc, bias_desc;

    cufftHandle full_fwd_plan;
    cufftHandle full_bwd_plan;
    cufftHandle single_fwd_plan;

    size_t workspace_size_ = 0;

    vec3i cs;

    float * ones;

private:
    void gemv( int m, int n, float alpha,
               const float *A, const float *x,
               float beta, float *y ) const
    {
        checkCublasErrors( cublasSgemv(cublas_handle_, CUBLAS_OP_N, m, n,
                                       &alpha, A, m, x, 1, &beta, y, 1) );
    }

public:
    device_array<float> forward( device_array<float> in )
    {
        long_t transform_elements = cs[0] * cs[1] * cs[2];
        long_t transform_bytes = transform_elements * sizeof(cuComplex);

        // Transform all the inputs
        auto in_t = get_device_array<cuComplex>
            (transform_elements * batch_size * num_inputs);

        checkCUFFT( cufftExecR2C(full_fwd_plan, in.get(), in_t.get()) );

        // We don't need the inputs anymore - save GPU memory
        in.reset();

        // We will store all the transforms here
        auto out_t = get_device_array<cuComplex>
            (transform_elements * batch_size * num_outputs);

        // Two scratch pads
        //
        // 1) Expanded kernels -> scratch1
        // 2) Transform scratch1 into scratch2
        // 3) For all appropriate inputs
        //     a) scratch1 = inputs * scratch2
        //     b) out = sum(sctach[i])

        {
            auto scratch1 = get_device_array<cuComplex>
                (transform_elements * num_inputs);
            auto scratch2 = get_device_array<cuComplex>
                (transform_elements * num_inputs);

            auto exploder_scratch = get_device_array<int>
                (kernel_len * num_inputs);

            kernel_exploder kexploder( exploder_scratch.get(),
                                       kernel_size,
                                       in_image_size,
                                       num_inputs );

            for ( long_t i = 0; i < num_outputs; ++i )
            {
                // explode kernel to kscratch
                kexploder.explode( kernels.get() + i * num_inputs * kernel_len,
                                   reinterpret_cast<float*>(scratch1.get()) );

                // fft of the kernels
                checkCUFFT( cufftExecR2C(single_fwd_plan,
                                         reinterpret_cast<float*>
                                         (scratch1.get()),
                                         scratch2.get()) );


                // Use the FFT(kernels for all the batches)
                for ( long_t j = 0; j < batch_size; ++j )
                {
                    long_t inumel = transform_elements * num_inputs ;
                    long_t onumel = transform_elements * num_outputs;

                    mul_add( scratch2.get(), scratch2.get() + inumel,
                             in_t.get() + j * inumel, scratch1.get() );


                    gemv(transform_elements * 2, num_inputs,
                         1, reinterpret_cast<float*>(scratch1.get()),
                         ones,
                         0, reinterpret_cast<float*>
                         (out_t.get() + j*onumel + i*transform_elements));

                    // for ( long_t k = 0; k < num_inputs; ++k )
                    // {
                    //     add_to( scratch1.get() + k * transform_elements,
                    //             scratch1.get() + (k+1) * transform_elements,
                    //             out_t.get() + j*onumel + i*transform_elements,
                    //             ( k == 0 ) ? 0 : 1 );
                    // }
                }
            }
        }


        in_t.reset();

        auto padded_out = get_device_array<float>
            ( batch_size * num_outputs * in_image_len );

        checkCUFFT( cufftExecC2R( full_bwd_plan, out_t.get(), padded_out.get()));

        // free some memory

        auto imploder_scratch = get_device_array<int>( out_image_len );

        image_imploder imploder( imploder_scratch.get(),
                                 in_image_size, kernel_size );

        auto out = get_device_array<float>(total_output_len);

        for ( long_t i = 0; i < num_outputs * batch_size; ++i )
        {
            imploder.implode( padded_out.get() + i * in_image_len,
                              out.get() + i * out_image_len );

        }

        div_all_by( out.get(), out.get() + total_output_len, in_image_len );

        float alpha = 1; float beta = 1;

        checkCUDNN(
            cudnnAddTensor( handle_,
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


    ~cufft_convolutional_layer()
    {
        checkCublasErrors( cublasDestroy(cublas_handle_) );

        checkCUDNN( cudnnDestroyTensorDescriptor(out_desc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(bias_desc) );

        cufftDestroy(full_fwd_plan);
        cufftDestroy(full_bwd_plan);
        cufftDestroy(single_fwd_plan);

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
    cufft_convolutional_layer( cudnnHandle_t& cudnn_handle,
                               long_t n, long_t fin, long_t fout,
                               vec3i const & is, vec3i const & ks,
                               float* km = nullptr, float* bs = nullptr )
        : convolutional_layer_base(n,fin,fout,is,ks)
        , handle_(cudnn_handle)
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

        checkCublasErrors( cublasCreate(&cublas_handle_) );

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

        cs = is;
        cs[2] /= 2; cs[2] += 1;

        // fft plans
        {
            int dims[3] = { static_cast<int>(is[0]),
                            static_cast<int>(is[1]),
                            static_cast<int>(is[2]) };

            int howmany = static_cast<int>(fin*fout*n);
            int rdist   = static_cast<int>(is[0]*is[1]*is[2]);
            int cdist   = static_cast<int>(is[0]*is[1]*(is[2]/2+1));

            checkCUFFT( cufftPlanMany(&full_fwd_plan, 3, dims, NULL, 0,
                                      rdist, NULL, 0,
                                      cdist, CUFFT_R2C,
                                      static_cast<int>(fin*n)) );

            checkCUFFT( cufftPlanMany(&full_bwd_plan, 3, dims, NULL, 0,
                                      cdist, NULL, 0,
                                      rdist, CUFFT_C2R,
                                      static_cast<int>(fout*n)) );

            checkCUFFT( cufftPlanMany(&single_fwd_plan, 3, dims, NULL, 0,
                                      rdist, NULL, 0,
                                      cdist, CUFFT_R2C,
                                      static_cast<int>(fin)) );

        }
    }
};



}}} // namespace znn::fwd::gpu
