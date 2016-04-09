#pragma once

#include "znn/host/v1/mfp.hpp"
#include "znn/host/v1/dp_fft_conv.hpp"
#include "znn/host/v1/fft_conv.hpp"
#include "znn/util/network.hpp"

namespace znn { namespace fwd { namespace host { namespace v1 {

class network
{
private:
    std::vector<std::unique_ptr<host::v1::host_layer>> layers_;

    vec5i  in_shape_  ;
    vec5i  out_shape_ ;

    long_t memory_required_ = 0;

public:
    long_t memory_required() const
    {
        return memory_required_;
    }

    template<typename ND>
    network( ND const & nd )
        : layers_(nd.layers().size())
        , in_shape_(nd.in_shape())
        , out_shape_(nd.out_shape())
    {
        for ( long_t i = 0; i < layers_.size(); ++i )
        {
            auto l = nd.layers()[i];

            if ( l.type == layer_type::convolutional )
            {
                if ( i == 0 )
                {
                    layers_[i]
                        = make_unique<host::v1::dp_fft_conv>
                        (l.batch_size,
                         l.num_inputs,
                         l.num_outputs,
                         l.in_image_size,
                         l.k_or_w_size,
                         l.hkernels->data(),
                         l.hbiases->data());
                }
                else
                {
                    layers_[i]
                        = make_unique<host::v1::fft_conv>
                        (l.batch_size,
                         l.num_inputs,
                         l.num_outputs,
                         l.in_image_size,
                         l.k_or_w_size,
                         l.hkernels->data(),
                         l.hbiases->data());
                }
            }
            else
            {
                layers_[i]
                    = make_unique<host::v1::mfp>
                    (l.batch_size,
                     l.num_inputs,
                     l.in_image_size,
                     l.k_or_w_size);
            }
            memory_required_ = std::max(memory_required_,
                                        layers_[i]->working_memory());

        }

        memory_required_ += nd.stored_memory();
    }

    vec5i const & in_shape() const
    {
        return in_shape_;
    }

    vec5i const & out_shape() const
    {
        return out_shape_;
    }

    host_tensor<real,5> forward( host_tensor<real,5> in )
    {
        for ( auto const & l: layers_ )
        {
            in = l->forward(std::move(in));
        }
        return in;
    }
};


}}}} // namespace znn::fwd::host::v1
