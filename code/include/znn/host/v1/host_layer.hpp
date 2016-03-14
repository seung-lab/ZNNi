#pragma once

#include "znn/types.hpp"
#include "znn/layer.hpp"
#include "znn/tensor/tensor.hpp"

namespace znn { namespace fwd { namespace host { namespace v1 {

class host_layer: public layer
{
public:
    using layer::layer;
    virtual host_tensor<float,5> forward( host_tensor<float,5> ) = 0;

    virtual long_t resident_memory() const = 0;
    virtual long_t working_memory() const = 0;

    virtual long_t total_memory() const
    {
        return this->resident_memory() + this->working_memory();
    }
};


}}}} // namespace znn::fwd::host::v1
