#pragma once

#include <iostream>
#include "znn/host/common/fft/optimal.hpp"
#include "znn/assert.hpp"
#include "znn/types.hpp"

namespace znn { namespace fwd { namespace host {

class padded_pruned_fft2d_transformer_base
{
protected:
    vec2i isize ; // image to be convolved size
    vec2i ksize ; // kernel size
    vec2i rsize ; // result size = isize - kernel + 1
    vec2i asize ; // actual transform size (to be padded to)
    vec2i csize ; // transform size

    real scale;

    long_t padded_len;

protected:
    padded_pruned_fft2d_transformer_base( vec2i const & is, vec2i const & ks )
        : isize(is)
        , ksize(ks)
        , rsize(is+vec2i::one-ks)
    {
        STRONG_ASSERT(is[0] >= ks[0]);
        STRONG_ASSERT(is[1] >= ks[1]);

        asize = get_optimal_size(isize);
        csize = asize;
        csize[0] /= 2; csize[0] += 1;
        scale = asize[0] * asize[1];

        padded_len = csize[0] * csize[1];
        padded_len += 7;
        padded_len -= (padded_len % 8);
    }

public:
    long_t p_len() const
    {
        return padded_len;
    }

    bool needs_padding() const
    {
        return isize != asize;
    }

    real get_scale() const
    {
        return scale;
    }

    vec2i const & image_size() const
    {
        return isize;
    }

    vec2i const & kernel_size() const
    {
        return ksize;
    }

    vec2i const & result_size() const
    {
        return rsize;
    }

    vec2i const & actual_size() const
    {
        return asize;
    }

    vec2i const & transform_size() const
    {
        return csize;
    }

    long_t image_elements() const
    {
        return isize[0] * isize[1];
    }

    long_t kernel_elements() const
    {
        return ksize[0] * ksize[1];
    }

    long_t result_elements() const
    {
        return rsize[0] * rsize[1];
    }

    long_t image_scratch_elements() const
    {
        return asize[0] * isize[1];
    }

    long_t kernel_scratch_elements() const
    {
        return asize[0] * ksize[1];
    }

    long_t result_scratch_elements() const
    {
        return asize[0] * rsize[1];
    }

    long_t actual_elements() const
    {
        return asize[0] * asize[1];
    }

    long_t transform_elements() const
    {
        return csize[0] * csize[1];
    }

    long_t image_memory() const
    {
        return isize[0] * isize[1] * sizeof(real);
    }

    long_t kernel_memory() const
    {
        return ksize[0] * ksize[1] * sizeof(real);
    }

    long_t result_memory() const
    {
        return rsize[0] * rsize[1] * sizeof(real);
    }

    long_t image_scratch_memory() const
    {
        return asize[0] * isize[1] * sizeof(real);
    }

    long_t kernel_scratch_memory() const
    {
        return asize[0] * ksize[1] * sizeof(real);
    }

    long_t result_scratch_memory() const
    {
        return asize[0] * rsize[1] * sizeof(real);
    }

    long_t transform_memory() const
    {
        return csize[0] * csize[1] * sizeof(real) * 2;
    }

    long_t result_offset() const
    {
        return (ksize[0]-1) * rsize[1];
    }
};

}}} // namespace znn::fwd::host
