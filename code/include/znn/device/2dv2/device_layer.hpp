#pragma once

#include "znn/types.hpp"
#include "znn/layer2d.hpp"
#include "znn/tensor/tensor.hpp"

#include <zi/time.hpp>

namespace znn { namespace fwd { namespace device { namespace twod {

class device_layer2d: public layer2d
{
public:
    using layer2d::layer2d;
    //virtual device_tensor<float,4> forward( device_tensor<float,4> ) const = 0;

    virtual void forward( float*, float*, void* ) const = 0;

    virtual long_t resident_memory() const = 0;
    virtual long_t working_memory() const = 0;

    virtual long_t total_memory() const
    {
        return this->resident_memory() + this->working_memory();
    }

    virtual long_t workspace_size() const
    {
        return 0;
    }

    std::pair<bool,double> workable() const
    {
        zi::wall_timer wt;
        try
        {
            device_tensor<float,4> in(rand_init,input_shape);
            device_tensor<float,4> out(rand_init,output_shape);
            device_array<char> ws(this->workspace_size());
            wt.reset();
            this->forward(in.data(), out.data(), ws.data());
            return std::make_pair(true, wt.elapsed<double>());
        }
        catch (...)
        {
            return std::make_pair(false,static_cast<double>(0));
        }
    }
};


}}}} // namespace znn::fwd::device::twod
