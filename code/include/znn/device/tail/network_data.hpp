#pragma once

#include "znn/assert.hpp"
#include "znn/util/network.hpp"
#include "znn/tensor/tensor.hpp"

#include <memory>

namespace znn { namespace fwd { namespace device { namespace v2 {

class network_data
{
public:
    struct layer_data
    {
        layer_type type;

        long_t batch_size    ;
        long_t num_inputs    ;
        long_t num_outputs   ;
        vec3i  in_image_size ;
        vec3i  k_or_w_size   ;

        std::shared_ptr<device_tensor<float,5>> kernels;
        std::shared_ptr<device_array<float>>    biases ;
    };

private:
    std::vector<layer_data> layers_    ;
    long_t                  batch_size_;
    long_t                  delta_in_  ;
    long_t                  delta_out_ ;
    vec5i                   in_shape_  ;
    vec5i                   out_shape_ ;

private:
    network_data( network_data const & other, long_t bs )
        : layers_(other.layers_)
        , batch_size_(bs)
        , delta_in_(other.delta_in_ * bs / other.batch_size_)
        , delta_out_(other.delta_out_ * bs / other.batch_size_)
        , in_shape_(other.in_shape())
        , out_shape_(other.out_shape())
    {
        STRONG_ASSERT(bs > 0);
        STRONG_ASSERT(bs <= other.batch_size_);

        in_shape_[0] *= bs;
        in_shape_[0] /= other.batch_size_;

        out_shape_[0] *= bs;
        out_shape_[0] /= other.batch_size_;

        for ( auto & l: layers_ )
        {
            l.batch_size *= bs;
            l.batch_size /= other.batch_size_;
        }
    }

public:

    std::vector<layer_data> const & layers() const
    {
        return layers_;
    }

    template<class Iter>
    network_data( Iter first, Iter last, long_t b, vec3i const & os )
        : layers_()
        , batch_size_(b)
    {
        vec3i is = first->fov + os - vec3i::one;

        delta_in_ = is[0] * is[1] * is[2] * b * first->num_inputs;

        in_shape_ = vec5i(b,first->num_inputs,is[0],is[1],is[2]);

        for ( auto l = first; l != last; ++l )
        {
            auto kw = l->k_or_w_size;

            if ( l->type == layer_type::convolutional )
            {
                layers_.push_back({ layer_type::convolutional,
                            b, l->num_inputs, l->num_outputs, is, kw,
                            std::make_shared<device_tensor<float,5>>(
                                rand_init, l->num_outputs, l->num_inputs,
                                kw[0], kw[1], kw[2]),
                            std::make_shared<device_array<float>>(
                                rand_init, l->num_outputs)});
                is = is - l->k_or_w_size + vec3i::one;
            }
            else
            {
                layers_.push_back({ layer_type::pooling,
                            b, l->num_inputs, l->num_outputs, is, kw,
                            nullptr, nullptr});

                long_t n = l->k_or_w_size[0]*l->k_or_w_size[1]*l->k_or_w_size[2];
                b *= n;
                is /= l->k_or_w_size;
            }

            delta_out_ = is[0] * is[1] * is[2] * b * l->num_outputs;
            out_shape_ = vec5i(b, l->num_outputs, is[0], is[1], is[2]);
        }
    }

    network_data( network_descriptor const & nd, long_t b, vec3i const & os )
        : network_data(nd.layers().begin(), nd.layers().end(), b, os)
    {}

    network_data fraction( long_t bs ) const
    {
        return network_data(*this, bs);
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
