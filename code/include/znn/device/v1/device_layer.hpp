#pragma once

#include "znn/types.hpp"
#include "znn/layer.hpp"
#include "znn/tensor/tensor.hpp"

namespace znn { namespace fwd { namespace device { namespace v1 {

class device_layer: public layer
{
public:
    using layer::layer;
    virtual device_tensor<float,5> forward( device_tensor<float,5> ) const = 0;

    virtual long_t resident_memory() const = 0;
    virtual long_t working_memory() const = 0;

    virtual long_t total_memory() const
    {
        return this->resident_memory() + this->working_memory();
    }

    bool workable() const
    {
        try
        {
            device_tensor<float,5> in(output_shape);
            auto out = this->forward(std::move(in));
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
};


}}}} // namespace znn::fwd::device::v1
