#pragma once

#include "types.hpp"
#include "memory.hpp"
#include "assert.hpp"

namespace znn { namespace fwd {

class convolutional_layer_base
{
public:
    long_t const batch_size ;
    long_t const num_inputs ;
    long_t const num_outputs;

    vec3i  const in_image_size  ;
    vec3i  const kernel_size    ;
    vec3i  const out_image_size ;

    long_t const in_image_len ;
    long_t const kernel_len   ;
    long_t const out_image_len;

    long_t const input_len   ;
    long_t const kernels_len ;
    long_t const output_len  ;

    long_t const total_input_len ;
    long_t const total_output_len;

    long_t const input_memory ;
    long_t const output_memory;
    long_t const kernel_memory;

public:
    convolutional_layer_base( long_t n, long_t fin, long_t fout,
                              vec3i const & is, vec3i const & ks ) noexcept
        : batch_size(n)
        , num_inputs(fin)
        , num_outputs(fout)
        , in_image_size(is)
        , kernel_size(ks)
        , out_image_size(is-ks+vec3i::one)
        , in_image_len(is[0]*is[1]*is[2])
        , kernel_len(ks[0]*ks[1]*ks[2])
        , out_image_len(out_image_size[0]*out_image_size[1]*out_image_size[2])
        , input_len(fin*in_image_len)
        , kernels_len(fin*fout*kernel_len)
        , output_len(out_image_len*fout)
        , total_input_len(input_len*n)
        , total_output_len(output_len*n)
        , input_memory(total_input_len*sizeof(real))
        , output_memory(total_output_len*sizeof(real))
        , kernel_memory(kernels_len*sizeof(real))
    { }
};


class pooling_layer_base
{
public:
    long_t const in_batch_size;
    long_t const num_inputs   ;
    long_t const num_outputs  ;

    vec3i  const in_image_size  ;
    vec3i  const window_size    ;
    vec3i  const out_image_size ;

    long_t const in_image_len   ;
    long_t const window_len     ;
    long_t const out_image_len  ;

    long_t const input_len   ;
    long_t const output_len  ;

    long_t const out_batch_size  ;
    long_t const total_input_len ;
    long_t const total_output_len;

    long_t const input_memory ;
    long_t const output_memory;

public:
    pooling_layer_base( long_t n, long_t fin,
                        vec3i const & is,
                        vec3i const & ws )
        : in_batch_size(n)
        , num_inputs(fin)
        , num_outputs(fin)
        , in_image_size(is)
        , window_size(ws)
        , out_image_size(is/ws)
        , in_image_len(is[0]*is[1]*is[2])
        , window_len(ws[0]*ws[1]*ws[2])
        , out_image_len(out_image_size[0]*out_image_size[1]*out_image_size[2])
        , input_len(fin*in_image_len)
        , output_len(out_image_len*fin)
        , out_batch_size(window_len*n)
        , total_input_len(input_len*n)
        , total_output_len(output_len*out_batch_size)
        , input_memory(total_input_len*sizeof(real))
        , output_memory(total_output_len*sizeof(real))
    {
        STRONG_ASSERT( (is+vec3i::one) % ws == vec3i::zero );
    }
};


}} // namespace znn::fwd
