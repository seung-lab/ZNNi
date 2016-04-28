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

enum struct layer2d_type
{
    convolutional, pooling, maxout
};

struct layer2d_descriptor
{
    layer2d_type type         ;
    long_t       num_inputs   ;
    long_t       num_outputs  ;
    vec2i        k_or_w_size  ;
    vec2i        fov          ;
};

class network2d_descriptor
{
private:
    std::vector<layer2d_descriptor> layers_;
    vec2i                           fov_ = vec2i::one;
    vec2i                           fragmentation_ = vec2i::one;

public:
    std::vector<layer2d_descriptor> const & layers() const
    {
        return layers_;
    }

    size_t num_layers() const
    {
        return layers_.size();
    }

    vec2i const & fov() const
    {
        return fov_;
    }

    vec2i const & fragmentation() const
    {
        return fragmentation_;
    }

    network2d_descriptor( std::string const & fname )
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
                long_t x, y;
                STRONG_ASSERT( std::fscanf(f, "%ld,%ld", &x, &y) == 2 );
                vec2i k_or_w(x,y);

                long_t w;
                STRONG_ASSERT( std::fscanf(f, "%ld\n", &w) == 1 );

                layers_.push_back({layer2d_type::convolutional, n_in, w,
                            k_or_w, vec2i::zero});

                std::cout << "CONV LAYER: " << k_or_w
                          << " :: " << n_in << " -> " << w << "\n";

                n_in = w;
            }
            else if ( std::string(nt) == "POOL" )
            {
                long_t x, y;
                STRONG_ASSERT( std::fscanf(f, "%ld,%ld", &x, &y) == 2 );
                vec2i k_or_w(x,y);

                layers_.push_back({layer2d_type::pooling, n_in, n_in,
                            k_or_w, vec2i::zero});

                std::cout << "POOL LAYER: " << k_or_w
                          << " :: " << n_in << " -> " << n_in << "\n";

            }
            else if ( std::string(nt) == "MAXOUT" )
            {
                long_t d;
                STRONG_ASSERT( std::fscanf(f, "%ld", &d) == 1 );
                vec2i k_or_w(0,0);

                layers_.push_back({layer2d_type::maxout, n_in, n_in/d,
                            k_or_w, vec2i::zero});

                std::cout << "MAXOUT LAYER: "
                          << " :: " << n_in << " -> " << (n_in/d) << "\n";

            }
            else
            {
                DIE("UNKNOWN LAYER TYPE");
            }
        }

        std::reverse( layers_.begin(), layers_.end() );

        for ( auto & l: layers_ )
        {
            if ( l.type == layer2d_type::convolutional )
            {
                fov_ += l.k_or_w_size - vec2i::one;
            }
            else if ( l.type == layer2d_type::pooling )
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


class znni_network2d
{
public:
    struct znni_layer
    {
        layer2d_descriptor descriptor;
        long_t             batch_size;
        vec2i              in_size   ;

        znni_layer( layer2d_descriptor const & ld, long_t b, vec2i const & s )
            : descriptor(ld)
            , batch_size(b)
            , in_size(s)
        {}


        host_tensor<real,4> random_kernels() const
        {
            return host_tensor<real,4>
                (rand_init,descriptor.num_outputs,descriptor.num_inputs,
                 descriptor.k_or_w_size[0],
                 descriptor.k_or_w_size[1]);
        }

        host_tensor<real,1> random_biases() const
        {
            return host_tensor<real,1>(rand_init,descriptor.num_outputs);
        }

        host_tensor<real,4> get_random_sample() const
        {
            return host_tensor<real,4>(rand_init,batch_size,
                                       descriptor.num_inputs,
                                       in_size[0], in_size[1]);
        }
    };

private:
    std::vector<znni_layer> layers_    ;
    vec2i                   in_size_   ;
    vec2i                   out_size_  ;
    vec4i                   in_shape_  ;
    long_t                  out_voxels_;

public:
    vec2i const & get_in_size() const
    {
        return in_size_;
    }

    vec2i const & get_out_size() const
    {
        return out_size_;
    }

    host_tensor<real,4> get_random_sample() const
    {
        return host_tensor<real,4>(rand_init,in_shape_);
    }

    std::vector<znni_layer> const & layers() const
    {
        return layers_;
    }

    vec4i in_shape() const
    {
        return in_shape_;
    }

    long_t out_voxels() const
    {
        return out_voxels_;
    }

    znni_network2d( network2d_descriptor const & nd,
                    long_t b, vec2i const & os )
        : layers_()
        , in_size_(os + nd.fov() - vec2i::one)
        , out_size_(os)
        , in_shape_(b,nd.layers().front().num_inputs,
                    in_size_[0], in_size_[1])
    {
        vec2i is = os + nd.fov() - vec2i::one;

        out_voxels_ = b * os[0] * os[1];

        for ( auto const & l: nd.layers() )
        {
            znni_layer znnil(l,b,is);

            if ( l.type == layer2d_type::convolutional )
            {
                is = is - l.k_or_w_size + vec2i::one;
            }
            else if ( l.type == layer2d_type::pooling)
            {
                long_t n = l.k_or_w_size[0]*l.k_or_w_size[1];
                b *= n;
                is /= l.k_or_w_size;
            }

            layers_.push_back(std::move(znnil));
        }

    }


};


}} // namespace znn::fwd
