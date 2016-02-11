#pragma once

#include "padded_pruned_fft.hpp"
#include "padded_pruned_parallel_fft.hpp"

namespace znn { namespace fwd { namespace cpu {


class padded_pruned_fft_auto_convolutional_layer
    : public cpu_convolutional_layer_base
    , public host_layer
{
private:
    std::unique_ptr<host_layer> layer_;

public:

    padded_pruned_fft_auto_convolutional_layer( task_package& handle,
                                           long_t n, long_t fin, long_t fout,
                                                vec3i const & is, vec3i const & ks,
                                                real * km = nullptr,
                                                real* bs = nullptr )
        : cpu_convolutional_layer_base( n, fin, fout, is, ks, km, bs )
    {
        if ( (n == 1) && (fin == 1) )
        {
            layer_ = std::unique_ptr<host_layer>
                ( new padded_pruned_parallel_fft_convolutional_layer
                  ( handle, n, fin, fout, is, ks, km, bs));
        }
        else
        {
            layer_ = std::unique_ptr<host_layer>
                ( new padded_pruned_fft_convolutional_layer
                  ( handle, n, fin, fout, is, ks, km, bs));
        }
    }

    host_array<real> forward( host_array<real> in ) const override
    {
        return layer_->forward(std::move(in));
    }

};



}}} // namespace znn::fwd::cpu
