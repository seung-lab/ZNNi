#pragma once

#include "znn/types.hpp"
#include "znn/layer.hpp"
#include "znn/tensor/tensor.hpp"

#include <zi/time.hpp>

namespace znn { namespace fwd { namespace device { namespace v2 {

class device_layer: public layer
{
public:
    using layer::layer;

    virtual void forward( float*, float*, void*, float ) const = 0;

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
            device_tensor<float,5> out(rand_init,output_shape);
            device_array<char> ws(this->workspace_size());
            wt.reset();
            this->forward(in.data(), out.data(), ws.data(), 0);
            return std::make_pair(true, wt.elapsed<double>());
        }
        catch (...)
        {
            return std::make_pair(false,static_cast<double>(0));
        }
    }
};


}}}} // namespace znn::fwd::device::v2
