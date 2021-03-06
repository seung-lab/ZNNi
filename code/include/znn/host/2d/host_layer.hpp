#pragma once

#include "znn/types.hpp"
#include "znn/layer2d.hpp"
#include "znn/tensor/tensor.hpp"

namespace znn { namespace fwd { namespace host { namespace twod {

class host_layer2d: public layer2d
{
public:
    using layer2d::layer2d;
    virtual host_tensor<float,4> forward( host_tensor<float,4> ) const = 0;

    virtual long_t resident_memory() const = 0;
    virtual long_t working_memory() const = 0;

    virtual long_t total_memory() const
    {
        return this->resident_memory() + this->working_memory();
    }
};


}}}} // namespace znn::fwd::host::twod
