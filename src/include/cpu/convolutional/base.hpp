#pragma once

#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../memory.hpp"
#include "../../layer.hpp"

namespace znn { namespace fwd { namespace cpu {

class cpu_convolutional_layer_base: public convolutional_layer_base
{
public:
    host_array<real> kernels  ;
    host_array<real> biases   ;

public:
    real* get_biases() const noexcept
    {
        return biases.get();
    }

    real* get_kernels() const noexcept
    {
        return kernels.get();
    }

public:
    cpu_convolutional_layer_base( long_t n, long_t fin, long_t fout,
                                  vec3i const & is, vec3i const & ks,
                                  real * km = nullptr, real* bs = nullptr )
        : convolutional_layer_base(n,fin,fout,is,ks)
        , kernels(get_array<real>(kernels_len))
        , biases(get_array<real>(fout))
    {
        if ( km )
        {
            std::copy_n(km, kernels_len, kernels.get());
        }
        if ( bs )
        {
            std::copy_n(bs, fout, biases.get());
        }
    }
};

}}} // namespace znn::fwd::cpu
