#pragma once

#include "in_split_conv3d.hpp"

namespace znn { namespace fwd { namespace gpu3dram {

class in_out_split_conv3d
{
private:
    long_t n_full_      ;
    long_t partial_size_;

    long_t in_stride_    ;
    long_t kernel_stride_;
    long_t bias_stride_  ;

    long_t workspace_memory_ = 0;

    in_split_conv3d* full_    = nullptr;
    in_split_conv3d* partial_ = nullptr;

public:
    long_t workspace_memory() const
    {
        return workspace_memory_;
    }

    void forward( float* in,
                  float* out,
                  float* kernels,
                  float* biases,
                  float* workspace_d) const
    {
        for ( long_t i = 0; i < n_full_; ++i )
        {
            full_->forward(in, out, kernels, biases, workspace_d);

            in      += in_stride_    ;
            kernels += kernel_stride_;
            biases  += bias_stride_  ;
        }

        if ( partial_size_ )
        {
            partial_->forward(in, out, kernels, biases, workspace_d);
        }
    }

public:
    ~in_out_split_conv3d()
    {
        delete full_;
        if ( partial_size_ ) delete partial_;
    }

    in_out_split_conv3d( cudnnHandle_t& handle,
                         long_t fin, long_t fin_chunk,
                         long_t fout, long_t fout_chunk,
                         vec3i const & is,
                         vec3i const & fs )
        : n_full_(fout/fout_chunk)
        , partial_size_(fout%fout_chunk)
    {
        in_stride_     = is[0] * is[1] * is[2] * fin * fout_chunk;
        kernel_stride_ = fs[0] * fs[1] * fs[2] * fin * fout_chunk;
        bias_stride_   = fout_chunk;

        full_ = new in_split_conv3d(handle, fin, fin_chunk, fout_chunk, is, fs);
        workspace_memory_ = full_->workspace_memory();

        if ( partial_size_ )
        {
            partial_ = new in_split_conv3d( handle,
                                            fin, fin_chunk, partial_size_,
                                            is, fs);
            workspace_memory_ = std::max(workspace_memory_,
                                         partial_->workspace_memory());
        }
    }
};


}}} // namespace znn::fwd::gpu3dram
