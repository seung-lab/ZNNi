#pragma once

#include "base.hpp"
#include "native_conv3d.hpp"


namespace znn { namespace fwd { namespace gpu3dram {

class batch_split_conv3d: public base_conv3d
{
private:
    long_t n_full_      ;
    long_t partial_size_;

    long_t workspace_memory_ = 0;

    native_conv3d full_   ;
    native_conv3d partial_;

public:
    ~batch_split_conv3d() {}

    void forward( float * in,
                  float * out,
                  float * kernels,
                  float * biases ) const override
    {
        float * workspace ;
        float * in_d      ;
        float * out_d     ;
        float * kernels_d ;
        float * biases_d  ;

        if ( workspace_memory_ )
        {
            checkCudaErrors( cudaMalloc(&workspace, workspace_memory_ ));
        }

        checkCudaErrors( cudaMalloc(&in_d     , full_->in_memory()     ));
        checkCudaErrors( cudaMalloc(&out_d    , full_->out_memory()    ));
        checkCudaErrors( cudaMalloc(&kernels_d, full_->kernel_memory() ));
        checkCudaErrors( cudaMalloc(&biases_d , full_->bias_memory()   ));

        checkCudaErrors( cudaMemcpy(kernels_d, kernels, full_->kernel_memory(),
                                    cudaMemcpyHostToDevice));
        checkCudaErrors( cudaMemcpy(biases_d, biases, full_->bias_memory(),
                                    cudaMemcpyHostToDevice));

        for ( long_t i = 0; i < n_full_, ++i )
        {
            checkCudaErrors( cudaMemcpy(in_d, in, full_->in_memory(),
                                        cudaMemcpyHostToDevice) );

            full_->forward(in_d, out_d, kernels_d, 0, workspace);
            full_->nonlinearity(out_d, biases_d);

            checkCudaErrors( cudaMemcpy(out, out_d, full_->out_memory(),
                                        cudaMemcpyDeviceToHost) );

            in  += full_->in_memory()  / sizeof(float);
            out += full_->out_memory() / sizeof(float);
        }

        if ( partial_size_ )
        {
            checkCudaErrors( cudaMemcpy(in_d, in, partial_->in_memory(),
                                        cudaMemcpyHostToDevice) );

            partial_->forward(in_d, out_d, kernels_d, 0, workspace);
            partial_->nonlinearity(out_d, biases_d);

            checkCudaErrors( cudaMemcpy(out, out_d, partial_->out_memory(),
                                        cudaMemcpyDeviceToHost) );
        }

        if ( full_->workspace_memory() )
        {
            checkCudaErrors( cudaMalloc(&workspace,
                                        full_->workspace_memory()));
        }

        full_->forward(in_d, out_d, kernels_d, 0, workspace);
        full_->nonlinearity(out_d, biases_d);

        if ( workspace_memory_ )
        {
            checkCudaErrors( cudaFree(workspace) );
        }

        checkCudaErrors( cudaFree(in_d));
        checkCudaErrors( cudaFree(out_d));
        checkCudaErrors( cudaFree(kernels_d));
        checkCudaErrors( cudaFree(biases_d));

    }


public:
    batch_split_conv3d( cudnnHandle_t& cudnn_handle,
                        long_t n,
                        long_t n_chunk,
                        long_t fin,
                        long_t fout,
                        vec3i const & is,
                        vec3i const & fs )
        : n_full_(n/n_chunk)
        , partial_size_(n%n_chunk)
    {
        full_ = new native_conv3d(cudnn_handle, n_chunk, fin, fout, is, fs);
        workspace_memory_ = full_->workspace_memory();

        if ( partial_size_ )
        {
            partial_ = new native_conv3d(cudnn_handle, has_partial_,
                                         fin, fout, is, fs);
            workspace_memory_ = std::max(workspace_memory_,
                                         partial_->workspace_memory());
        }
    }

};

}}} // namespace znn::fwd::gpu3dram
