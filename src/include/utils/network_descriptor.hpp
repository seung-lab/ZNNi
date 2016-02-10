#pragma once

#include "../types.hpp"
#include "../assert.hpp"
#include "../memory.hpp"
#include "../init.hpp"

#include <vector>
#include <iostream>
#include <list>
#include <string>
#include <cstdio>
#include <cstdlib>


namespace znn { namespace fwd {

enum struct layer_type
{
    convolutional, pooling
};

struct layer_descriptor
{
    layer_type type         ;
    long_t     num_inputs   ;
    long_t     num_outputs  ;
    vec3i      k_or_w_size  ;
};

class network_descriptor
{
private:
    std::list<layer_descriptor> layers_;
    vec3i                       fov_ = vec3i::one;

public:
    std::list<layer_descriptor> const & layers() const
    {
        return layers_;
    }

    vec3i const & fov() const
    {
        return fov_;
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
            long_t x, y, z;

            STRONG_ASSERT( std::fscanf(f, "%s %ld,%ld,%ld",
                                       nt, &x, &y, &z) == 4 );

            vec3i k_or_w(x,y,z);

            if ( std::string(nt) == "CONV" )
            {
                long_t w;
                STRONG_ASSERT( std::fscanf(f, "%ld\n", &w) == 1 );

                layers_.push_front({layer_type::convolutional, n_in, w, k_or_w});

                std::cout << "CONV LAYER: " << k_or_w
                          << " :: " << n_in << " -> " << w << "\n";

                n_in = w;
            }
            else if ( std::string(nt) == "POOL" )
            {
                layers_.push_front({layer_type::pooling, n_in, n_in, k_or_w});

                std::cout << "POOL LAYER: " << k_or_w
                          << " :: " << n_in << " -> " << n_in << "\n";

            }
            else
            {
                DIE("UNKNOWN LAYER TYPE");
            }
        }

        for ( auto & l: layers_ )
        {
            if ( l.type == layer_type::convolutional )
            {
                fov_ += l.k_or_w_size - vec3i::one;
            }
            else
            {
                fov_ *= l.k_or_w_size;
            }
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
        host_array<real> kernels   ;
        host_array<real> biases    ;

        znni_layer( layer_descriptor const & ld, long_t b, vec3i const & s )
            : descriptor(ld)
            , batch_size(b)
            , in_size(s)
        {}
    };

private:
    std::vector<znni_layer> layers_    ;
    long_t                  batch_size_;
    vec3i                   in_size_   ;
    vec3i                   out_size_  ;
    uniform_init            init_      ;
    long_t                  in_len_    ;
    long_t                  out_len_   ;

public:
    long_t get_batch_size() const
    {
        return batch_size_;
    }

    vec3i const & get_in_size() const
    {
        return in_size_;
    }

    vec3i const & get_out_size() const
    {
        return in_size_;
    }

    host_array<real> get_random_sample()
    {
        host_array<real> r = get_array<real>(in_len_);
        init_.initialize(r.get(), in_len_);
        return r;
    }

    std::vector<znni_layer> const & layers() const
    {
        return layers_;
    }

    long_t in_len() const
    {
        return in_len_;
    }

    long_t out_len() const
    {
        return out_len_;
    }


    znni_network( network_descriptor const & nd,
                  long_t b, vec3i const & os )
        : layers_()
        , batch_size_(b)
        , in_size_(os + nd.fov() - vec3i::one)
        , out_size_(os)
        , init_(0.1)
        , in_len_(in_size_[0]*in_size_[1]*in_size_[2]*b)
    {
        vec3i is = os + nd.fov() - vec3i::one;

        out_len_ = b * os[0] * os[1] * os[2];

        for ( auto const & l: nd.layers() )
        {
            znni_layer znnil(l,b,is);

            if ( l.type == layer_type::convolutional )
            {
                long_t klen = l.k_or_w_size[0] * l.k_or_w_size[1]
                    * l.k_or_w_size[2] * l.num_inputs * l.num_outputs;
                long_t blen = l.num_outputs;

                znnil.kernels = get_array<real>(klen);
                znnil.biases  = get_array<real>(blen);

                init_.initialize( znnil.kernels.get(), klen );
                init_.initialize( znnil.biases.get() , blen );
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

}} // namespace znn::fwd
