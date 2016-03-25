#pragma once

#include "znn/device/v2/device_network_executor.hpp"
#include "znn/device/v2/network_data.hpp"

namespace znn { namespace fwd { namespace device { namespace v2 {

class best_device_network
{
private:
    vec5i  in_shape_  ;
    vec5i  out_shape_ ;

    std::unique_ptr<device_network_executor> full_ = nullptr;
    std::unique_ptr<device_network_executor> part_ = nullptr;

    long_t n_full_ = 0;

public:

    best_device_network( network_data const & nd )
        : in_shape_(nd.in_shape())
        , out_shape_(nd.out_shape())
    {
        double best = std::numeric_limits<double>::max();

        for ( long_t b = 1; b <= nd.batch_size(); ++b )
        {
            std::cout << "Trying best device net for b = " << b << std::endl;
            try
            {
                auto full = make_unique<device_network_executor>
                    (nd.fraction(b));

                long_t n_full = nd.batch_size() / b;
                double time = full->benchmark() * n_full;

                long_t part_len = nd.batch_size() % b;

                std::unique_ptr<device_network_executor> part = nullptr;

                if ( part_len )
                {
                    part = make_unique<device_network_executor>
                        (nd.fraction(part_len));

                    time += part->benchmark();
                }

                if ( time < best )
                {
                    best = time;
                    full_ = std::move(full);
                    part_ = std::move(part);
                    n_full_ = n_full;
                }
            }
            catch (...)
            {
            }
        }

        if ( !full_ )
        {
            throw std::logic_error("not feasable");
        }
    }

    host_tensor<float,5> forward( host_tensor<float,5> in ) const
    {
        host_tensor<float,5> ret(out_shape_);

        float const * inp  = in.data();
        float       * outp = ret.data();

        for ( long_t i = 0; i < n_full_; ++i )
        {
            full_->forward(inp,outp);
            inp += full_->delta_in();
            outp += full_->delta_out();
        }

        if ( part_ )
        {
            part_->forward(inp,outp);
        }

        return ret;
    }

    double benchmark() const
    {
        zi::wall_timer wt;
        host_tensor<float,5> in(rand_init,in_shape_);
        wt.reset();
        in = forward(std::move(in));
        return wt.elapsed<double>();
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
