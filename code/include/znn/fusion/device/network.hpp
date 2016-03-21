#pragma once

#include "znn/fusion/device/sub_network.hpp"

namespace znn { namespace fwd { namespace device { namespace fusion {

class network
{
private:
    vec5i  in_shape_  ;
    vec5i  out_shape_ ;

    std::unique_ptr<sub_network> full_ = nullptr;
    std::unique_ptr<sub_network> part_ = nullptr;

    long_t n_full_ = 1;

    double estimate_ = 0;

private:
    template<typename ND>
    void search( long_t low, long_t high, ND const & nd, bool b )
    {
        for ( long_t mid = low + 1; mid <= high; ++mid )
        {

            LOG(network) << "search ("
                         << low << "," << high << ")"
                         << " :: " << mid;

            try
            {
                auto full = sub_network::get(nd.fraction(mid), b);

                if ( !full )
                {
                    continue;
                }

                long_t n_full = nd.batch_size() / mid;

                double time = full->estimate_runtime() * n_full;

                long_t part_len = nd.batch_size() % mid;

                std::unique_ptr<sub_network> part = nullptr;

                if ( part_len )
                {
                    part = sub_network::get(nd.fraction(part_len), b);
                    if ( !part )
                    {
                        continue;
                    }
                    time += part->estimate_runtime();
                }

                LOG(network) << "search filed for "
                             << mid << " : " << time << " was " << estimate_;

                if ( time < estimate_ )
                {
                    full_ = std::move(full);
                    part_ = std::move(part);
                    n_full_ = n_full;
                    estimate_ = time;
                }

            }
            catch ( std::exception & e )
            {
                LOG(network) << "search filed for "
                             << mid << " : " << e.what();
            }
        }
    }

public:
    template<typename ND>
    network( ND const & nd, bool b )
        : in_shape_(nd.in_shape())
        , out_shape_(nd.out_shape())
    {
        full_ = sub_network::get(nd.fraction(1), b);
        n_full_ = nd.batch_size();

        if ( !full_ )
        {
            throw std::logic_error("no feasable network");
        }

        estimate_ = full_->estimate_runtime() * nd.batch_size();

        search(1, nd.batch_size(), nd, b);
    }

public:
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

    double estimate_runtime() const
    {
        zi::wall_timer wt;
        host_tensor<float,5> in(rand_init,in_shape_);
        wt.reset();
        in = forward(std::move(in));
        return wt.elapsed<double>();
    }

    double benchmark( long_t n ) const
    {
        double time = 0;
        for ( long_t i = 0; i < n; ++i )
        {
            time += estimate_runtime();
        }
        return time / n;
    }

    vec5i const & in_shape() const
    {
        return in_shape_;
    }

    vec5i const & out_shape() const
    {
        return out_shape_;
    }

    std::string name() const
    {
        std::ostringstream oss;
        oss << "full(" << n_full_ << "): {";
        oss << full_->name();
        if ( part_ )
        {
            oss << "} part: {" << part_->name();
        }
        oss << "}";
        return oss.str();
    }

    template<typename ND>
    static std::unique_ptr<network> get( ND const & nd, bool b )
    {
        std::unique_ptr<network> ret;
        try
        {
            ret = make_unique<network>(nd, b);
        }
        catch ( std::exception & e )
        {
            LOG(network) << "failed to get network: " << e.what();
        }

        return ret;
    }

};


}}}} // namespace znn::fwd::device::fusion
