#pragma once

#include "base.hpp"
#include "in_out_split_conv3d.hpp"

namespace znn { namespace fwd { namespace gpu3dram {


class full_split_conv3d: public base_conv3d
{
private:
    long_t delta_in_ ;
    long_t delta_out_;
    long_t n_        ;

    in_out_split_conv3d* impl_ = nullptr;

public:

    void forward( float* in,
                  float* out,
                  float* kernels,
                  float* biases) const override
    {
        float * workspace_d;

        if ( impl_->workspace_memory() )
        {
            checkCudaErrors( cudaMalloc(&workspace_d,
                                        impl_->workspace_memory()));
        }

        for ( long_t i = 0; i < n_; ++i )
        {
            impl_->forward(in, out, kernels, biases, workspace_d);
            in  += delta_in_ ;
            out += delta_out_;
        }

        if ( impl_->workspace_memory() )
        {
            checkCudaErrors( cudaFree(workspace_d) );
        }
    }

public:
    ~full_split_conv3d()
    {
        delete impl_;
    }

    full_split_conv3d( cudnnHandle_t& handle,
                       long_t n,
                       long_t fin, long_t fin_chunk,
                       long_t fout, long_t fout_chunk,
                       vec3i const & is,
                       vec3i const & fs )
        : n_(n)
    {
        impl_ = new in_out_split_conv3d( handle,
                                         fin, fin_chunk,
                                         fout, fout_chunk,
                                         is, fs);

        vec3i os = is + vec3i::one - fs;

        delta_in_  = fin  * is[0] * is[1] * is[2];
        delta_out_ = fout * os[0] * os[1] * os[2];
    }
};




}}} // namespace znn::fwd::gpu3dram
