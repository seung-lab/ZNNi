#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"
#include "znn/tensor/tensor.hpp"

#include <vector>
#include <iostream>
#include <list>
#include <string>
#include <cstdio>
#include <cstdlib>


namespace znn { namespace fwd {

enum struct layer_type
{
    convolutional, pooling, maxout
};

struct layer_descriptor
{
    layer_type type         ;
    long_t     num_inputs   ;
    long_t     num_outputs  ;
    vec3i      k_or_w_size  ;
    vec3i      fov          ;
};

class network_descriptor
{
private:
    std::vector<layer_descriptor> layers_;
    vec3i                         fov_ = vec3i::one;
    vec3i                         fragmentation_ = vec3i::one;

public:
    std::vector<layer_descriptor> const & layers() const
    {
        return layers_;
    }

    size_t num_layers() const
    {
        return layers_.size();
    }

    vec3i const & fov() const
    {
        return fov_;
    }

    vec3i const & fragmentation() const
    {
        return fragmentation_;
    }

    network_descriptor( std::string const & fname )
        : layers_()
    {
        FILE* f = std::fopen(fname.c_str(), "r");
        long_t n_in, n_l;

        STRONG_ASSERT( std::fscanf(f, "INPUTS:%ld, LAYERS:%ld\n",
                                   &n_in, &n_l) == 2 );

        std::cout << "Network with " << n_in << " inputs and "
                  << n_l << " layers\n";


        for ( long_t i = 0; i < n_l; ++i )
        {
            char nt[1024];
            STRONG_ASSERT( std::fscanf(f, "%s", nt) == 1 );

            if ( std::string(nt) == "CONV" )
            {
                long_t x, y, z;
                STRONG_ASSERT( std::fscanf(f, "%ld,%ld,%ld", &x, &y, &z) == 3 );
                vec3i k_or_w(x,y,z);

                long_t w;
                STRONG_ASSERT( std::fscanf(f, "%ld\n", &w) == 1 );

                layers_.push_back({layer_type::convolutional, n_in, w,
                            k_or_w, vec3i::zero});

                std::cout << "CONV LAYER: " << k_or_w
                          << " :: " << n_in << " -> " << w << "\n";

                n_in = w;
            }
            else if ( std::string(nt) == "POOL" )
            {
                long_t x, y, z;
                STRONG_ASSERT( std::fscanf(f, "%ld,%ld,%ld", &x, &y, &z) == 3 );
                vec3i k_or_w(x,y,z);

                layers_.push_back({layer_type::pooling, n_in, n_in,
                            k_or_w, vec3i::zero});

                std::cout << "POOL LAYER: " << k_or_w
                          << " :: " << n_in << " -> " << n_in << "\n";

            }
            else if ( std::string(nt) == "MAXOUT" )
            {
                long_t d;
                STRONG_ASSERT( std::fscanf(f, "%ld", &d) == 1 );
                vec3i k_or_w(0,0,0);

                layers_.push_back({layer_type::maxout, n_in, n_in/d,
                            k_or_w, vec3i::zero});

                std::cout << "MAXOUT LAYER: "
                          << " :: " << n_in << " -> " << (n_in/d) << "\n";

                n_in /= d;
            }
            else
            {
                DIE("UNKNOWN LAYER TYPE");
            }
        }

        std::reverse( layers_.begin(), layers_.end() );

        for ( auto & l: layers_ )
        {
            if ( l.type == layer_type::convolutional )
            {
                fov_ += l.k_or_w_size - vec3i::one;
            }
            else if ( l.type == layer_type::pooling )
            {
                fov_ *= l.k_or_w_size;
                fragmentation_ *= l.k_or_w_size;
            }
            l.fov = fov_;
        }

        std::reverse(layers_.begin(), layers_.end());

        std::cout << "   FOV: " << fov_ << "\n";
    }
};


class znni_network
{
public:
    struct znni_layer
    {
        layer_descriptor descriptor;
        long_t           batch_size;
        vec3i            in_size   ;

        znni_layer( layer_descriptor const & ld, long_t b, vec3i const & s )
            : descriptor(ld)
            , batch_size(b)
            , in_size(s)
        {}


        host_tensor<real,5> random_kernels() const
        {
            return host_tensor<real,5>
                (rand_init,descriptor.num_outputs,descriptor.num_inputs,
                 descriptor.k_or_w_size[0],
                 descriptor.k_or_w_size[1],
                 descriptor.k_or_w_size[2]);
        }

        host_tensor<real,1> random_biases() const
        {
            return host_tensor<real,1>(rand_init,descriptor.num_outputs);
        }

        host_tensor<real,5> get_random_sample() const
        {
            return host_tensor<real,5>(rand_init,batch_size,
                                       descriptor.num_inputs,
                                       in_size[0], in_size[1], in_size[2]);
        }
    };

private:
    std::vector<znni_layer> layers_    ;
    vec3i                   in_size_   ;
    vec3i                   out_size_  ;
    vec5i                   in_shape_  ;
    long_t                  out_voxels_;

public:
    vec3i const & get_in_size() const
    {
        return in_size_;
    }

    vec3i const & get_out_size() const
    {
        return out_size_;
    }

    host_tensor<real,5> get_random_sample() const
    {
        return host_tensor<real,5>(rand_init,in_shape_);
    }

    std::vector<znni_layer> const & layers() const
    {
        return layers_;
    }

    vec5i in_shape() const
    {
        return in_shape_;
    }

    long_t out_voxels() const
    {
        return out_voxels_;
    }

    znni_network( network_descriptor const & nd,
                  long_t b, vec3i const & os )
        : layers_()
        , in_size_(os + nd.fov() - vec3i::one)
        , out_size_(os)
        , in_shape_(b,nd.layers().front().num_inputs,
                    in_size_[0], in_size_[1], in_size_[2])
    {
        vec3i is = os + nd.fov() - vec3i::one;

        out_voxels_ = b * os[0] * os[1] * os[2];

        for ( auto const & l: nd.layers() )
        {
            znni_layer znnil(l,b,is);

            if ( l.type == layer_type::convolutional )
            {
                is = is - l.k_or_w_size + vec3i::one;
            }
            else
            {
                long_t n = l.k_or_w_size[0]*l.k_or_w_size[1]*l.k_or_w_size[2];
                b *= n;
                is /= l.k_or_w_size;
            }

            layers_.push_back(std::move(znnil));
        }

    }


};


class znni_pooling_network
{
public:
    using znni_layer = znni_network::znni_layer;
private:
    std::vector<znni_layer> layers_    ;
    vec3i                   in_size_   ;
    vec3i                   out_size_  ;
    vec5i                   in_shape_  ;
    long_t                  out_voxels_;

public:
    vec3i const & get_in_size() const
    {
        return in_size_;
    }

    vec3i const & get_out_size() const
    {
        return out_size_;
    }

    host_tensor<real,5> get_random_sample() const
    {
        return host_tensor<real,5>(rand_init,in_shape_);
    }

    std::vector<znni_layer> const & layers() const
    {
        return layers_;
    }

    vec5i in_shape() const
    {
        return in_shape_;
    }

    long_t out_voxels() const
    {
        return out_voxels_;
    }

    znni_pooling_network( network_descriptor const & nd,
                          long_t b, vec3i const & os )
        : layers_()
        , in_size_(nd.fov() + (os - vec3i::one) * nd.fragmentation())
        , out_size_(os)
        , in_shape_(b,nd.layers().front().num_inputs,
                    in_size_[0], in_size_[1], in_size_[2])
    {
        vec3i is = nd.fov() + (os - vec3i::one) * nd.fragmentation();

        out_voxels_ = b * os[0] * os[1] * os[2];

        for ( auto const & l: nd.layers() )
        {
            znni_layer znnil(l,b,is);

            if ( l.type == layer_type::convolutional )
            {
                is = is - l.k_or_w_size + vec3i::one;
            }
            else
            {
                is /= l.k_or_w_size;
            }

            layers_.push_back(std::move(znnil));
        }

    }


};

}} // namespace znn::fwd
