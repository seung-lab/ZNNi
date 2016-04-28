#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"

namespace znn { namespace fwd { namespace host { namespace twod {

class conv_data2d
{
protected:
    host_tensor<float,4> kernels;
    host_tensor<float,1> biases ;

public:
    conv_data2d( long_t fin, long_t fout, vec2i const & ks,
                 float * km = nullptr, float* bs = nullptr )
        : kernels(fout,fin,ks[0],ks[1])
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


}}}} // namespace znn::fwd::host::twod
