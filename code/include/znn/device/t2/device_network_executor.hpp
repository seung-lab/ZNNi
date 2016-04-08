#pragma once

#include "znn/device/v2/best_device_layer.hpp"
#include "znn/device/v2/network_data.hpp"

namespace znn { namespace fwd { namespace device { namespace v2 {

class device_network_executor
{
private:
    std::vector<std::unique_ptr<device_layer>> layers;
    long_t batch_size_;
    long_t delta_in_  ;
    long_t delta_out_ ;
    vec5i  in_shape_  ;
    vec5i  out_shape_ ;

public:

    device_network_executor( network_data const & nd )
        : layers()
        , batch_size_(nd.batch_size())
        , delta_in_(nd.delta_in())
        , delta_out_(nd.delta_out())
        , in_shape_(nd.in_shape())
        , out_shape_(nd.out_shape())
    {
        for ( auto & l: nd.layers() )
        {
            std::cout << "Tried layer" << std::endl;
            auto best = get_best_device_layer(l);
            std::cout << "Tried layer " << best.get() << std::endl;
            if ( best )
            {
                layers.push_back(std::move(best));
            }
            else
            {
                throw std::logic_error("no layer feasable");
            }
        }
    }

    void forward( float const * in, float * out ) const
    {
        device_tensor<float,5> din(in_shape_);
        din.load(in, from_host);

        for ( auto & l: layers )
        {
            din = l->forward(std::move(din));
        }

        din.store(out, to_host);
    }

    double benchmark() const
    {
        zi::wall_timer wt;
        host_tensor<float,5> out(out_shape_);
        host_tensor<float,5> in(rand_init,in_shape_);
        wt.reset();
        forward(in.data(),out.data());
        return wt.elapsed<double>();
    }

    long_t batch_size() const
    {
        return batch_size_;
    }

    long_t delta_in() const
    {
        return delta_in_;
    }

    long_t delta_out() const
    {
        return delta_out_;
    }

    vec5i const & in_shape() const
    {
        return in_shape_;
    }

    vec5i const & out_shape() const
    {
        return out_shape_;
    }
};

}}}} // namespace znn::fwd::device::v2
