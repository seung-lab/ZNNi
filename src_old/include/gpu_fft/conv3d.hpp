#pragma once

#include <cufft.h>

#include "../gpu/cuda_utils.hpp"
#include "../types.hpp"
#include "../assert.hpp"
#include "utils.hpp"

namespace znn { namespace fwd { namespace gpu_fft {

class conv3d
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

    cufftHandle full_fwd_plan;
    cufftHandle full_bwd_plan;

    cufftHandle single_fwd_plan;
    cufftHandle single_bwd_plan;

    long_t n_;
    long_t fin_;
    long_t fout_;
    vec3i is_;
    vec3i fs_;
    vec3i os_;
    vec3i cs_;

private:


public:
    void forward( float * in,
                  float * out,
                  float * kernels,
                  float   beta,
                  void  * workspace ) const
    {
        long_t transform_elements = cs_[0] * cs_[1] * cs_[2];
        long_t transform_bytes = transform_elements * sizeof(cufftComplex);

        // fft of the inputs
        cuComplex* in_t
            = byte_offset<cuComplex>(workspace, n_*fout_*transform_bytes);

        checkCUFFT( cufftExecR2C(full_fwd_plan, in, in_t) );

        // kernel scratch
        float* kscratch
            = byte_offset<float>(in_t, fin_*transform_bytes);

        cuComplex* kscratch_t
            = byte_offset<cuComplex>(kscratch, fin_*transform_bytes);

        cuComplex* out_t = reinterpret_cast<cuComplex*>(workspace);

        cuComplex* product = reinterpret_cast<cuComplex*>(kscratch);

        kernel_exploder kexploder
            ( byte_offset<int>(kscratch_t, fin_*transform_bytes),
              fs_, is_, n_ );

        for ( long_t i = 0; i < fout_; ++i )
        {
            // explode kernel to kscratch
            kexploder.explode( kernels + i * fin_ * fs_[0] * fs_[1] * fs_[2],
                               kscratch )

            // fft of the kernels
            checkCUFFT( cufftExecR2C(single_fwd_plan, kscratch, kscratch_t) );

            cuComplex* in_ts  = in_t;
            cuComplex* out_ts = out_t + i * transform_elements;

            for ( long_t j = 0; j < n_; ++j )
            {
                mul_all( kscratch_t, kscratch_t + transform_elements * fin_,
                         in_ts, product );

                add_to( product, product + transform_elements,
                        out_ts )
                for ( long_t k = 0; k < fin_; ++k )
                {
                    add_to(
                }

                in_ts  += fin_  * transform_elements;
                out_ts += fout_ * transform_elements;
            }
        }


        // real domain workspace
        // complex domain workspace

        //cufftExecR2C(fwd_plan, in, cin);
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

    ~native_conv3d()
    {
        checkCUDNN( cudnnDestroyTensorDescriptor(in_desc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(out_desc) );

        checkCUDNN( cudnnDestroyTensorDescriptor(bias_desc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filter_desc) );

        cufftDestroy(full_fwd_plan);
        cufftDestroy(full_bwd_plan);
        cufftDestroy(single_fwd_plan);
        cufftDestroy(single_bwd_plan);
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
    native_conv3d( cudnnHandle_t& cudnn_handle,
                   long_t n,
                   long_t fin,
                   long_t fout,
                   vec3i const & is,
                   vec3i const & fs )
        : cudnn_handle_(cudnn_handle)
        , n_(n)
        , fin_(fin)
        , fout_(fout)
        , is_(is)
        , fs_(fs)
    {


        kernel_memory_ = fin * fout * fs[0] * fs[1] * fs[2] * sizeof(float);
        bias_memory_   = fout * sizeof(float);

        vec3i os = is + vec3i::one - fs;

        os_ = os;
        cs_ = is_; cs_[2] /= 2; cs_[2] += 1;

        create_tensor_descriptor(&in_desc,n,fin,is[0],is[1],is[2]);
        create_tensor_descriptor(&out_desc,n,fout,os[0],os[1],os[2]);
        create_tensor_descriptor(&bias_desc,1,fout,1,1,1);

        checkCUDNN( cudnnCreateFilterDescriptor(&filter_desc) );
        {
            int dims[5] = {static_cast<int>(fout),
                           static_cast<int>(fin),
                           static_cast<int>(fs[0]),
                           static_cast<int>(fs[1]),
                           static_cast<int>(fs[2])};
            checkCUDNN(
                cudnnSetFilterNdDescriptor(filter_desc,
                                           CUDNN_DATA_FLOAT,
                                           5, dims));
        }

        // fft plans
        {
            int dims[3] = { static_cast<int>(os[0]),
                            static_cast<int>(os[1]),
                            static_cast<int>(os[2]) };

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

            checkCUFFT( cufftPlanMany(&single_bwd_plan, 3, dims, NULL, 0,
                                      cdist, NULL, 0,
                                      rdist, CUFFT_C2R,
                                      static_cast<int>(fout)) );


        }
        in_memory_  = n * fin  * is[0] * is[1] * is[2] * sizeof(float);
        out_memory_ = n * fout * os[0] * os[1] * os[2] * sizeof(float);
};


}}} // namespace znn::fwd::gpu3dram
