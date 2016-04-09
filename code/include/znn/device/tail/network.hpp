#pragma once

#include "znn/log.hpp"
#include "znn/device/tail/sub_network.hpp"

namespace znn { namespace fwd { namespace device { namespace tail {

template<typename Conv>
class network
{
private:
    typedef sub_network<Conv> sub_net;

private:
    vec5i  in_shape_  ;
    vec5i  out_shape_ ;

    std::unique_ptr<sub_net> full_ = nullptr;
    std::unique_ptr<sub_net> part_ = nullptr;

    long_t n_full_ = 1;

private:
    template<typename ND>
    void binary_search( long_t low, long_t high, ND const & nd )
    {
        long_t mid = (low + high) / 2;

        LOG(network) << "binary_search ("
                     << low << "," << high << ")"
                     << " :: " << mid;

        if ( low == mid )
        {
            return;
        }

        auto full = sub_net::get(nd.fraction(mid));

        if ( !full || !full->feasable() )
        {
            binary_search( low, mid, nd );
            return;
        }

        long_t n_full = nd.batch_size() / mid;

        long_t part_len = nd.batch_size() % mid;
        std::unique_ptr<sub_net> part = nullptr;

        if ( part_len )
        {
            part = sub_net::get(nd.fraction(part_len));
            if ( !part || !part->feasable() )
            {
                binary_search( low, mid, nd );
                return;
            }
        }

        full_ = std::move(full);
        part_ = std::move(part);
        n_full_ = n_full;

        binary_search( mid, high, nd );
    }

public:
    template<typename ND>
    network( ND const & nd )
        : in_shape_(nd.in_shape())
        , out_shape_(nd.out_shape())
    {
        full_ = sub_net::get(nd);
        n_full_ = 1; //nd.batch_size();

        if ( !full_ || !full_->feasable() )
        {
            if ( nd.batch_size() == 1 )
            {
                throw std::logic_error("no feasable network");
            }

            full_ = sub_net::get(nd.fraction(1));
            n_full_ = nd.batch_size();

            if ( !full_ || !full_->feasable() )
            {
                throw std::logic_error("no feasable network");
            }

            binary_search(1, nd.batch_size(), nd);
        }
    }

    long_t memory_requirement() const
    {
        return full_->memory_requirement();
    }

    long_t working_memory() const
    {
        long_t a = in_shape_[0] * in_shape_[1] * in_shape_[2]
            * in_shape_[3] * in_shape_[4];
        long_t b = out_shape_[0] * out_shape_[1] * out_shape_[2]
            * out_shape_[3] * out_shape_[4];
        return (a+b) * sizeof(float);
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
            inp  += full_->delta_in();
            outp += full_->delta_out();
        }

        if ( part_ )
        {
            part_->forward(inp,outp);
        }

        return ret;
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
    static std::unique_ptr<network> get( ND const & nd )
    {
        std::unique_ptr<network> ret;
        try
        {
            ret = make_unique<network>(nd);
        }
        catch ( std::exception & e )
        {
            LOG(network) << "failed to get network: " << e.what();
        }

        return ret;
    }

};


}}}} // namespace znn::fwd::device::tail
