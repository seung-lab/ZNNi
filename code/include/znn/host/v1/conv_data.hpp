#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"

namespace znn { namespace fwd { namespace host { namespace v1 {

class conv_data
{
protected:
    host_tensor<float,5> kernels;
    host_tensor<float,1> biases ;

public:
    conv_data( long_t fin, long_t fout, vec3i const & ks,
               float * km = nullptr, float* bs = nullptr )
        : kernels(fout,fin,ks[0],ks[1],ks[2])
        , biases(fout)
    {
        if ( km )
            kernels.load(km, from_host);
        else
            kernels.randomize();

        if ( bs )
            biases.load(bs, from_host);
        else
            biases.randomize();
    }
};


}}}} // namespace znn::fwd::host::v1
