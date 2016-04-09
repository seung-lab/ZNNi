#pragma once

#include "znn/log.hpp"
#include "znn/util/network.hpp"
#include "znn/device/tail/cudnn_conv.hpp"
#include "znn/device/tail/fft_conv.hpp"
#include "znn/device/tail/mfp.hpp"

#include <sstream>

namespace znn { namespace fwd { namespace device { namespace tail {

template<typename Conv>
class sub_network
{
private:
    std::vector<std::unique_ptr<tail_layer>> layers;
    long_t batch_size_;
    long_t delta_in_  ;
    long_t delta_out_ ;
    vec5i  in_shape_  ;
    vec5i  out_shape_ ;

private:
    template<typename T>
    std::unique_ptr<tail_layer> get_layer( T const & l )
    {
        std::unique_ptr<tail_layer> ret = nullptr;

        if ( l.type == layer_type::convolutional )
        {
            try
            {
                ret = make_unique<Conv>(l.batch_size,
                                        l.num_inputs,
                                        l.num_outputs,
                                        l.in_image_size,
                                        l.k_or_w_size,
                                        l.dkernels,
                                        l.dbiases);
            }
            catch ( std::exception & e )
            {
                LOG(sub_network::get_layer) << "Conv failed: " << e.what();
            }
        }
        else
        {
            try
            {
                ret = make_unique<mfp>(l.batch_size,
                                       l.num_inputs,
                                       l.in_image_size,
                                       l.k_or_w_size);

            }
            catch ( std::exception & e )
            {
                LOG(sub_network::get_layer) << "mfp failed: " << e.what();
            }
        }

        return ret;
    }


public:
    template<typename ND>
    sub_network( ND const & nd )
        : layers()
        , batch_size_(nd.batch_size())
        , delta_in_(nd.delta_in())
        , delta_out_(nd.delta_out())
        , in_shape_(nd.in_shape())
        , out_shape_(nd.out_shape())
    {
        for ( auto & l: nd.layers() )
        {
            auto lr = get_layer(l);
            if ( lr )
            {
                layers.push_back(std::move(lr));
            }
            else
            {
                throw std::logic_error("no feasable sub_network");
            }
        }
    }

public:
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

    long_t memory_requirement() const
    {
        long_t resident = 0;
        long_t working  = 0;

        for ( auto & l: layers )
        {
            resident += l->resident_memory();
            working = std::max(working, l->working_memory());
        }

        return working + resident;
    }

    bool feasable() const
    {
        return (memory_requirement() / 1024 / 1024 / 1024) <= 9;
    }

    template<typename ND>
    static std::unique_ptr<sub_network> get( ND const & nd )
    {
        std::unique_ptr<sub_network> ret;
        try
        {
            ret = make_unique<sub_network>(nd);
        }
        catch ( std::exception & e )
        {
            LOG(sub_network) << "failed to get sub_network: " << e.what();
        }

        return ret;
    }

    std::string name() const
    {
        std::ostringstream oss;
        oss << "layers: ";
        for ( auto & l: layers )
        {
            oss << ' ' << l->name();
        }
        return oss.str();
    }
};

}}}} // namespace znn::fwd::device::tail
