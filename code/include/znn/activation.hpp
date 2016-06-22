#pragma once

#include "znn/types.hpp"


namespace znn { namespace fwd {

enum class activation
{
    none, logistics, sigmoid, relu, tanh, clipped_relu
};


void relu(real* out, long_t out_image_len)
{
    for ( long_t i = 0; i < out_image_len; ++i )
    {
        out[i] = std::max(static_cast<real>(0), out[i]);
    }
}

void logistics(real* out, long_t out_image_len )
{
    for ( long_t i = 0; i<out_image_len; ++i)
        out[i] = static_cast<real>(1) / ((static_cast<real>(1)) + std::exp(-out[i]));
}


void activation_function(  real* out
                           , long_t out_image_len
                           , activation activation_=activation::none)
{
    // activation function
    switch (activation_)
    {
    case activation::none:
        break;
    case activation::sigmoid:
        logistics(out, out_image_len);
        break;
    case activation::logistics:
        logistics(out, out_image_len);
        break;
    case activation::relu:
        relu(out, out_image_len);
        break;
    default:
        DIE("unknown activation");
    }
}



}} // namespace znn::fwd
