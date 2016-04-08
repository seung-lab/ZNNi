#pragma once

#include "znn/assert.hpp"
#include "znn/util/network.hpp"
#include "znn/tensor/tensor.hpp"

#include <memory>

namespace znn { namespace fwd {

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

        std::shared_ptr<host_tensor<float,5>> hkernels;
        std::shared_ptr<host_array<float>>    hbiases ;

        std::shared_ptr<device_tensor<float,5>> dkernels;
        std::shared_ptr<device_array<float>>    dbiases ;
    };

private:
    std::vector<layer_data> layers_    ;
    long_t                  batch_size_;
    long_t                  delta_in_  ;
    long_t                  delta_out_ ;
    vec5i                   in_shape_  ;
    vec5i                   out_shape_ ;
    long_t                  stored_memory_ = 0;

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
            stored_memory_ += l.hkernels->num_elements();
            stored_memory_ += l.hbiases->num_elements();
        }

        stored_memory_ *= sizeof(float);
    }

    struct tail_tag {};

    network_data( network_data const & other, tail_tag )
        : layers_(0)
        , batch_size_(other.batch_size_)
        , delta_in_()
        , delta_out_(other.delta_out_)
        , in_shape_()
        , out_shape_(other.out_shape())
    {
        STRONG_ASSERT(other.layers_.size() > 1);

        batch_size_ = other.layers_[1].batch_size;

        in_shape_[0]  = other.layers_[1].batch_size;
        in_shape_[1]  = other.layers_[1].num_inputs;
        in_shape_[2]  = other.layers_[1].in_image_size[0];
        in_shape_[3]  = other.layers_[1].in_image_size[1];
        in_shape_[4]  = other.layers_[1].in_image_size[1];

        delta_in_ = in_shape_[0] * in_shape_[1] * in_shape_[2]
            * in_shape_[3] * in_shape_[4];

        layers_.resize(other.layers_.size()-1);

        for ( size_t i = 0; i < layers_.size(); ++i )
        {
            layers_[i] = other.layers_[i+1];
            stored_memory_ += layers_[i].hkernels->num_elements();
            stored_memory_ += layers_[i].hbiases->num_elements();
        }

        stored_memory_ *= sizeof(float);
    }


public:
    long_t stored_memory() const
    {
        return stored_memory_;
    }

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
                auto hkernels
                    = std::make_shared<host_tensor<float,5>>(
                        rand_init, l->num_outputs, l->num_inputs,
                        kw[0], kw[1], kw[2]);

                auto dkernels
                    = std::make_shared<device_tensor<float,5>>(
                        l->num_outputs, l->num_inputs,
                        kw[0], kw[1], kw[2]);

                (*dkernels) = (*hkernels);

                auto hbiases
                    = std::make_shared<host_array<float>>(
                        rand_init, l->num_outputs);

                auto dbiases
                    = std::make_shared<device_array<float>>(
                        l->num_outputs);

                (*dbiases) = (*hbiases);

                stored_memory_ += hkernels->num_elements();
                stored_memory_ += hbiases->num_elements();

                layers_.push_back({ layer_type::convolutional,
                            b, l->num_inputs, l->num_outputs, is, kw,
                            hkernels, hbiases, dkernels, dbiases});
                is = is - l->k_or_w_size + vec3i::one;
            }
            else
            {
                layers_.push_back({ layer_type::pooling,
                            b, l->num_inputs, l->num_outputs, is, kw,
                            nullptr, nullptr, nullptr, nullptr});

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

    network_data( network_data const & other )
        : layers_(other.layers_)
        , batch_size_(other.batch_size_)
        , delta_in_(other.delta_in_)
        , delta_out_(other.delta_out_)
        , in_shape_(other.in_shape())
        , out_shape_(other.out_shape())
        , stored_memory_(other.stored_memory())
    {}

    network_data& operator =( network_data const & other )
    {
        layers_ = other.layers_;
        batch_size_  = other.batch_size_;
        delta_in_ = other.delta_in_;
        delta_out_ = other.delta_out_;
        in_shape_ = other.in_shape_;
        out_shape_ = other.out_shape_;
        stored_memory_ = other.stored_memory_;
        return *this;
    }

    network_data fraction( long_t bs ) const
    {
        return network_data(*this, bs);
    }

    network_data tail() const
    {
        return network_data(*this, tail_tag());
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


}} // namespace znn::fwd
