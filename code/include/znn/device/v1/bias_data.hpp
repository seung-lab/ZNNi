#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"

namespace znn { namespace fwd { namespace device { namespace v1 {

class bias_data
{
protected:
    device_tensor<float,1> biases ;

public:
    bias_data( long_t fout, float* bs = nullptr )
        : biases(fout)
    {
        if ( bs )
            biases.load(bs, from_host);
        else
            biases.randomize();
    }
};


}}}} // namespace znn::fwd::device::v1
