#pragma once

#include <cudnn.h>

#include "cuda_utils.hpp"
#include "../types.hpp"
#include "../assert.hpp"
#include "../init.hpp"

#include "simple_conv_layer.hpp"

namespace znn { namespace fwd { namespace gpu3dram {

class input_divided_layer
{
private:
    long_t fin_;
    long_t bin_;
    long_t fout_;

    long_t per_kernel_memory_;

    long_t workspace_memory_ = 0;
    long_t memory_;

    simple_conv_layer* full_ = nullptr;
    simple_conv_layer* rest_ = nullptr;

public:
    long_t workspace_memory() const override { workspace_memory_; }
    long_t memory() const override { memory_; }

    void forward(float* in,
                 float* out,
                 float* kernels,
                 float* biases,
                 float* dworkspace) const override
    {
        float * din;
        float * dout;
        float * dkernels;
        float * dbiases;

        checkCudaErrors( cudaMalloc(&din,      full_->in_memory()     ));
        checkCudaErrors( cudaMalloc(&dout,     full_->out_memory()    ));
        checkCudaErrors( cudaMalloc(&dkernels, full_->kernel_memory() ));
        checkCudaErrors( cudaMalloc(&dbiases,  full_->bias_memory()   ));

        checkCudaErrors( cudaMemcpy(dbiases,
                                    biases,
                                    full_->bias_memory(),
                                    cudaMemcpyHostToDevice) );

        long_t hkernel_stride = fin_ * per_kernel_memory_;
        long_t dkernel_stride = bin_ * per_kernel_memory_;

        for ( long_t i = 0; i < (fin_/bin_); ++i )
        {
            // copy the input
            checkCudaErrors( cudaMemcpy(din, in, full_->in_memory(),
                                        cudaMemcpyHostToDevice) );

            // copy the kernels
            {
                for ( long_t k = 0; k < fout_; ++k )
                {
                    checkCudaErrors(
                        cudaMemcpy(dkernels + k * dkernel_stride/sizeof(float),
                                   kernels  + k * hkernel_stride/sizeof(float),
                                   dkernel_stride,
                                   cudaMemcpyHostToDevice) );
                }
            }

            full_->forward(din, dout, dkernels, (i==0)?0:1, dworkspace);

            in      += full_->in_memory()        / sizeof(float);
            kernels += bin_ * per_kernel_memory_ / sizeof(float);
        }

        if ( (fin_ % din_) != 0 )
        {
            long_t dkernel_stride = (fin_%bin_) * per_kernel_memory_;

            // copy the input
            checkCudaErrors( cudaMemcpy(din, in, rest_->in_memory(),
                                        cudaMemcpyHostToDevice) );

            // copy the kernels
            {
                for ( long_t k = 0; k < fout_; ++k )
                {
                    checkCudaErrors(
                        cudaMemcpy(dkernels + k * dkernel_stride/sizeof(float),
                                   kernels  + k * hkernel_stride/sizeof(float),
                                   dkernel_stride,
                                   cudaMemcpyHostToDevice) );
                }
            }

            rest_->forward(din, dout, dkernels, 1, dworkspace);
        }

        full_->nonlinearity(dout, dbiases);
        checkCudaErrors( cudaMemcpy(out, dout, full_->out_memory(),
                                    cudaMemcpyDeviceToHost) );

        checkCudaErrors( cudaFree(din));
        checkCudaErrors( cudaFree(dout));
        checkCudaErrors( cudaFree(dkernels));
        checkCudaErrors( cudaFree(dbiases));
    }

public:
    ~input_divided_layer()
    {
        delete full_;
        if ( rest_ )
        {
            delete rest_;
        }
    }

    input_divided_layer( cudnnHandle_t& cudnn_handle,
                         long_t fin, long_t bin, long_t fout,
                         vec3i const & is,
                         vec3i const & fs )
        : fin_(fin)
        , bin_(bin)
        , fout_(fout)
    {
        per_kernel_memory_ = fs[0] * fs[1] * fs[2] * sizeof(float);

        full_ = new simple_conv_layer(cudnn_handle, 1, bin, fout, is, fs);
        workspace_memory_ = full_->workspace_memory();
        memory_ = full_->memory();

        if ( (fin_ % bin_) != 0 )
        {
            rest_ = new simple_conv_layer(cudnn_handle, 1, fin_%bin_,
                                          fout, is, fs);

            workspace_memory_ = std::max(workspace_memory_,
                                         rest_->workspace_memory());

            memory_ = std::max(memory_, rest_->memory());
        }
    }
};


class input_output_divided_layer
{
private:
    long_t fin_;
    long_t fout_;
    long_t bout_;

    long_t kernel_elements_;

    long_t workspace_memory_ = 0;
    long_t memory_;

    input_divided_layer* full_ = nullptr;
    input_divided_layer* rest_ = nullptr;

public:
    long_t workspace_memory() const override { workspace_memory_; }
    long_t memory() const override { memory_; }

    void forward(float* in,
                 float* out,
                 float* kernels,
                 float* biases,
                 float* dworkspace) const override
    {
        long_t kernel_stride = kernel_elements_ * fin_ * bout_;

        for ( long_t i = 0; i < (fout_/bout_); ++i )
        {
            full_->forward(in, out, kernels, biases, dworkspace);

            in      += full_->in_memory() / sizeof(float);
            kernels += kernel_elements_;
            biases  += bout_;
        }

        if ( (fout_ % bout_) != 0 )
        {
            rest_->forward(in, out, kernels, biases, dworkspace);
        }
    }

public:
    ~input_output_divided_layer()
    {
        delete full_;
        if ( rest_ )
        {
            delete rest_;
        }
    }

    input_output_divided_layer( cudnnHandle_t& cudnn_handle,
                                long_t fin, long_t bin,
                                long_t fout, long_t bout,
                                vec3i const & is,
                                vec3i const & fs )
        : fin_(fin)
        , fout_(fout)
        , bout_(out)
    {
        kernel_elements_ = fs[0] * fs[1] * fs[2];

        full_ = new input_divided_layer(cudnn_handle, fin, bin, bout, is, fs);
        workspace_memory_ = full_->workspace_memory();
        memory_ = full_->memory();

        if ( (fout % bout) != 0 )
        {
            rest_ = new input_divided_layer(cudnn_handle,
                                            fin, fout%bout,
                                            is, fs);

            workspace_memory_ = std::max(workspace_memory_,
                                         rest_->workspace_memory());

            memory_ = std::max(memory_, rest_->memory());
        }
    }
};


class batch_input_output_divided_layer
{
private:
    long_t delta_in_ ;
    long_t delta_out_;
    long_t n_        ;

    input_output_divided_layer* impl_ = nullptr;

public:
    long_t workspace_memory() const override { workspace_memory_; }
    long_t memory() const override { memory_; }

    void forward( float* in,
                  float* out,
                  float* kernels,
                  float* biases) const override
    {
        float * workspace;

        if ( impl_->workspace_memory() )
        {
            checkCudaErrors( cudaMalloc(&workspace, impl_->workspace_memory()));
        }

        for ( long_t i = 0; i < n_; ++i )
        {
            impl_->forward(in, out, kernels, biases, workspace);
            in  += delta_in_ ;
            out += delta_out_;
        }

        if ( impl_->workspace_memory() )
        {
            checkCudaErrors( cudaFree(workspace) );
        }
    }

public:
    ~batch_input_output_divided_layer()
    {
        delete impl_;
    }

    batch_input_output_divided_layer( cudnnHandle_t& cudnn_handle,
                                      long_t n,
                                      long_t fin, long_t bin,
                                      long_t fout, long_t bout,
                                      vec3i const & is,
                                      vec3i const & fs )
        : n_(n)
    {
        full_ = new input_output_divided_layer(cudnn_handle,
                                               fin, bin,
                                               fout, bout,
                                               is, fs);

        vec3i os = is + vec3i::one - fs;

        delta_in_  = fin  * is[0] * is[1] * is[2];
        delta_out_ = fout * os[0] * os[1] * os[2];
    }
};




}}} // namespace znn::fwd::gpu3dram
