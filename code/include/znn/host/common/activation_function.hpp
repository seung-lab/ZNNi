#pragma once

#include "znn/types.hpp"

namespace znn { namespace fwd { namespace host { namespace v1 {

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


void activation_function(real* out, long_t out_image_len, long_t act_func_type)
{
    if (act_func_type == 0)
        return;
    else if (act_func_type == 1)
        // rectified linear function
        relu(out, out_image_len);
    else if (act_func_type==2)
        //logistics linear function
        logistics(out, out_image_len);
    else
        throw std::invalid_argument("invalid activation function type!");
}


}}}} // namespace znn::fwd::host::v1
