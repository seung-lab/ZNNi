#pragma once

#include "../host_layer.hpp"
#include "../handle.hpp"
#include "padded_pruned_fft.hpp"
#include "padded_pruned_parallel_fft.hpp"

namespace znn { namespace fwd { namespace tbb {


class padded_pruned_fft_auto_convolutional_layer
    : public host_layer
{
private:
    std::unique_ptr<host_layer> layer_;

public:
    padded_pruned_fft_auto_convolutional_layer( void*,
                                                long_t n, long_t fin, long_t fout,
                                                vec3i const & is, vec3i const & ks,
                                                real * km = nullptr,
                                                real* bs = nullptr )
    {
        if ( (n == 1) && (fin == 1) )
        {
            layer_ = std::unique_ptr<host_layer>
                ( new padded_pruned_parallel_fft_convolutional_layer
                  ( (void*)0, n, fin, fout, is, ks, km, bs));
        }
        else
        {
            layer_ = std::unique_ptr<host_layer>
                ( new padded_pruned_fft_convolutional_layer
                  ( (void*)(0), n, fin, fout, is, ks, km, bs));
        }
    }

    host_array<real> forward( host_array<real> in ) const override
    {
        return layer_->forward(std::move(in));
    }

};



}}} // namespace znn::fwd::tbb
