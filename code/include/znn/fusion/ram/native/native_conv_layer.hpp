#pragma once

#include "znn/types.hpp"
#include "znn/layer.hpp"
#include "znn/tensor/tensor.hpp"

#include <zi/time.hpp>

namespace znn { namespace fwd { namespace device { namespace fusion {
namespace ram { namespace native {

class native_conv_layer: public conv_layer<layer>
{
private:
    typedef conv_layer<layer> super_type;
public:
    using super_type::super_type;

    virtual void forward( device_tensor<float,5>, float*,
                          device_tensor<float,5> const &,
                          float, void* ) const = 0;
    virtual char const * name() const = 0;
    virtual size_t workspace_size() const = 0;
    virtual void nonlineiaruty( float*, float const * ) const = 0;

    double estimate_runtime( long_t n ) const
    {
        std::cout << "herea " << this->workspace_size() << std::endl;
        device_array<char> ws(this->workspace_size());
        std::cout << "herez" << std::endl;
        device_tensor<float,5> in(rand_init,input_shape);
        std::cout << "herea" << std::endl;
        device_tensor<float,5> out(output_shape);
        std::cout << "herea" << std::endl;
        device_tensor<float,5> kernels(rand_init,kernels_shape);
        std::cout << "herea" << std::endl;

        zi::wall_timer wt;
        wt.reset();
        std::cout << "herea" << std::endl;
        this->forward(std::move(in), out.data(), kernels, 0, ws.data());
        std::cout << "hereb" << std::endl;
        double time = wt.elapsed<double>();

        if ( n > 1 )
        {
            device_tensor<float,5> in(rand_init,input_shape);
            wt.reset();
            this->forward(std::move(in), out.data(), kernels, 1, ws.data());
            time += wt.elapsed<double>();
        }

        return time;
    }

};

}}}}}} // namespace znn::fwd::device::fusion::ram::native
