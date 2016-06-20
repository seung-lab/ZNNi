#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"

namespace znn { namespace fwd { namespace device { namespace v2 {

class conv_data
{
protected:
    device_tensor<float,6> kernels;
    device_tensor<float,1> biases ;

public:
    conv_data( long_t n, long_t fin, long_t fout,
               vec3i const & ks,
               float * km = nullptr, float* bs = nullptr )
        : kernels(n,fout,fin,ks[0],ks[1],ks[2])
        , biases(fout * n)
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


}}}} // namespace znn::fwd::device::v2
