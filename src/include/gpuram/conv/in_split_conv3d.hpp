#pragma once

#include "native_conv3d.hpp"

namespace znn { namespace fwd { namespace gpu3dram {

class in_split_conv3d
{
private:
    long_t fout_        ;
    long_t n_full_      ;
    long_t partial_size_;

    long_t in_stride_     ;
    long_t hkernel_stride_;
    long_t dkernel_stride_;
    long_t partial_kernel_stride_ = 0;

    long_t workspace_memory_ = 0;

    native_conv3d* full_    = nullptr;
    native_conv3d* partial_ = nullptr;

public:
    long_t workspace_memory() const
    {
        return workspace_memory_;
    }

    void forward(float* in,
                 float* out,
                 float* kernels,
                 float* biases,
                 float* workspace_d) const
    {
        float * in_d      ;
        float * out_d     ;
        float * kernels_d ;
        float * biases_d  ;

        checkCudaErrors( cudaMalloc(&in_d     , full_->in_memory()     ));
        checkCudaErrors( cudaMalloc(&out_d    , full_->out_memory()    ));
        checkCudaErrors( cudaMalloc(&kernels_d, full_->kernel_memory() ));
        checkCudaErrors( cudaMalloc(&biases_d , full_->bias_memory()   ));

        checkCudaErrors( cudaMemcpy(biases_d, biases, full_->bias_memory(),
                                    cudaMemcpyHostToDevice) );

        for ( long_t i = 0; i < n_full_; ++i )
        {
            // copy the input
            checkCudaErrors( cudaMemcpy(in_d, in, full_->in_memory(),
                                        cudaMemcpyHostToDevice) );

            // copy the kernels
            {
                for ( long_t k = 0; k < fout_; ++k )
                {
                    checkCudaErrors(
                        cudaMemcpy(kernels_d + k * dkernel_stride_,
                                   kernels   + k * hkernel_stride_,
                                   dkernel_stride_ * sizeof(float),
                                   cudaMemcpyHostToDevice) );
                }
            }

            full_->forward(in_d, out_d, kernels_d, (i==0)?0:1, workspace_d);

            in      += in_stride_;
            kernels += dkernel_stride_;
        }

        if ( partial_size_ )
        {
            // copy the input
            checkCudaErrors( cudaMemcpy(in_d, in, partial_->in_memory(),
                                        cudaMemcpyHostToDevice) );

            // copy the kernels
            {
                for ( long_t k = 0; k < fout_; ++k )
                {
                    checkCudaErrors(
                        cudaMemcpy(kernels_d + k * partial_kernel_stride_,
                                   kernels   + k * hkernel_stride_,
                                   partial_kernel_stride_ * sizeof(float),
                                   cudaMemcpyHostToDevice) );
                }
            }

            partial_->forward(in_d, out_d, kernels_d, 1, workspace_d);
        }

        full_->nonlinearity(out_d, biases_d);

        checkCudaErrors( cudaMemcpy(out_d, out_d, full_->out_memory(),
                                    cudaMemcpyDeviceToHost) );

        checkCudaErrors( cudaFree(in_d));
        checkCudaErrors( cudaFree(out_d));
        checkCudaErrors( cudaFree(kernels_d));
        checkCudaErrors( cudaFree(biases_d));
    }

public:
    ~in_split_conv3d()
    {
        delete full_;
        if ( partial_size_ ) delete partial_;
    }

    in_split_conv3d( cudnnHandle_t& handle,
                     long_t fin, long_t fin_chunk, long_t fout,
                     vec3i const & is,
                     vec3i const & fs )
        : fout_(fout)
        , n_full_(fin/fin_chunk)
        , partial_size_(fin%fin_chunk)
    {
        in_stride_ = is[0] * is[1] * is[2] * fin;

        hkernel_stride_ = fs[0] * fs[1] * fs[2] * fin;
        dkernel_stride_ = fs[0] * fs[1] * fs[2] * fin_chunk;

        full_ = new native_conv3d(handle, 1, fin_chunk, fout, is, fs);
        workspace_memory_ = full_->workspace_memory();

        if ( partial_size_ )
        {
            partial_kernel_stride_ = fs[0] * fs[1] * fs[2] * partial_size_;
            partial_ = new native_conv3d(handle, 1, partial_size_,
                                         fout, is, fs);
            workspace_memory_ = std::max(workspace_memory_,
                                         partial_->workspace_memory());
        }
    }
};


}}} // namespace znn::fwd::gpu3dram
