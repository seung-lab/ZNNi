#pragma once

#include "znn/types.hpp"
#include "znn/layer.hpp"
#include "znn/tensor/tensor.hpp"

#include <zi/time.hpp>

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

    std::pair<bool,double> workable() const
    {
        zi::wall_timer wt;
        try
        {
            device_tensor<float,5> in(rand_init,input_shape);
            wt.reset();
            auto out = this->forward(std::move(in));
            return std::make_pair(true, wt.elapsed<double>());
        }
        catch (...)
        {
            return std::make_pair(false,static_cast<double>(0));
        }
    }
};


}}}} // namespace znn::fwd::device::v1
