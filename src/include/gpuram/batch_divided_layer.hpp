#pragma once

#include <cudnn.h>

#include "../types.hpp"
#include "../assert.hpp"
#include "../init.hpp"

#include "simple_conv_layer.hpp"

namespace znn { namespace fwd { namespace gpu3dram {

class batch_divided_layer
{
private:
    long_t n_;
    long_t b_;

    long_t workspace_memory_ = 0;
    long_t memory_;

    simple_conv_layer* full_ = nullptr;
    simple_conv_layer* rest_ = nullptr;

public:
    long_t workspace_memory() const { workspace_memory_; }
    long_t memory() const { memory_; }

    void forward(float* in,
                 float* out,
                 float* kernels,
                 float* biases ) const
    {
        float * workspace;
        float * din;
        float * dout;
        float * dkernels;
        float * dbiases;

        checkCudaErrors( cudaMalloc(&din,      full_->in_memory()     ));
        checkCudaErrors( cudaMalloc(&dout,     full_->out_memory()    ));
        checkCudaErrors( cudaMalloc(&dkernels, full_->kernel_memory() ));
        checkCudaErrors( cudaMalloc(&dbiases,  full_->bias_memory()   ));

        checkCudaErrors( cudaMemcpy(dkernels, kernels, full_->kernel_memory(),
                                    cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMemcpy(dbiases,  biases,  full_->bias_memory(),
                                    cudaMemcpyHostToDevice) );

        if ( workspace_memory_ )
        {
            checkCudaErrors( cudaMalloc(&workspace, workspace_memory_ ));
        }

        for ( long_t i = 0; i < (n_/b_); ++i )
        {
            checkCudaErrors( cudaMemcpy(din, in, full_->in_memory(),
                                        cudaMemcpyHostToDevice) );

            full_->forward(din, dout, dkernels, 0, workspace);
            full_->nonlinearity(dout, dbiases);

            checkCudaErrors( cudaMemcpy(out, dout, full_->out_memory(),
                                        cudaMemcpyDeviceToHost) );

            in  += full_->in_memory()  / sizeof(float);
            out += full_->out_memory() / sizeof(float);
        }

        if ( (n_ % b_) != 0 )
        {
            checkCudaErrors( cudaMemcpy(din, in, rest_->in_memory(),
                                        cudaMemcpyHostToDevice) );

            rest_->forward(din, dout, dkernels, 0, workspace);
            rest_->nonlinearity(dout, dbiases);

            checkCudaErrors( cudaMemcpy(out, dout, rest_->out_memory(),
                                        cudaMemcpyDeviceToHost) );

        }

        if ( workspace_memory_ )
        {
            checkCudaErrors( cudaFree(workspace) );
        }


        checkCudaErrors( cudaFree(din));
        checkCudaErrors( cudaFree(dout));
        checkCudaErrors( cudaFree(dkernels));
        checkCudaErrors( cudaFree(dbiases));

    }

public:
    ~batch_divided_layer()
    {
        delete full_;
        if ( rest_ )
        {
            delete rest_;
        }
    }

    batch_divided_layer( cudnnHandle_t& cudnn_handle,
                         long_t n, long_t b, long_t fin, long_t fout,
                         vec3i const & is,
                         vec3i const & fs )
        : n_(n)
        , b_(b)
    {
        full_ = new simple_conv_layer(cudnn_handle, b, fin, fout, is, fs);
        workspace_memory_ = full_->workspace_memory();
        memory_ = full_->memory();

        if ( (n % b) != 0 )
        {
            rest_ = new simple_conv_layer(cudnn_handle, n%b, fin, fout, is, fs);
            workspace_memory_ = std::max(workspace_memory_,
                                         rest_->workspace_memory());
            memory_ = std::max(memory_, rest_->memory());
        }
    }

};

}}} // namespace znn::fwd::gpu3dram
