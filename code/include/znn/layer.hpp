#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"

namespace znn { namespace fwd {

class layer
{
public:
    long_t const in_batch_size   ;
    long_t const num_inputs      ;
    vec3i  const in_image_size   ;
    long_t const in_image_len    ;
    long_t const input_len       ;
    long_t const total_input_len ;
    long_t const input_memory    ;
    vec5i  const input_shape     ;

    long_t const out_batch_size  ;
    long_t const num_outputs     ;
    vec3i  const out_image_size  ;
    long_t const out_image_len   ;
    long_t const output_len      ;
    long_t const total_output_len;
    long_t const output_memory   ;
    vec5i  const output_shape    ;

public:
    layer( long_t nin, long_t fin, vec3i const & is,
           long_t nout, long_t fout, vec3i const & os ) noexcept
        : in_batch_size(nin)
        , num_inputs(fin)
        , in_image_size(is)
        , in_image_len(is[0]*is[1]*is[2])
        , input_len(fin*in_image_len)
        , total_input_len(input_len*nin)
        , input_memory(total_input_len*sizeof(float))
        , input_shape(nin,fin,is[0],is[1],is[2])
        , out_batch_size(nout)
        , num_outputs(fout)
        , out_image_size(os)
        , out_image_len(os[0]*os[1]*os[2])
        , output_len(fout*out_image_len)
        , total_output_len(output_len*nout)
        , output_memory(total_output_len*sizeof(float))
        , output_shape(nout,fout,os[0],os[1],os[2])
    { }
};

template<typename Base>
class conv_layer: public Base
{
public:
    vec3i  const kernel_size   ;
    long_t const kernel_len    ;
    long_t const kernels_len   ;
    long_t const kernels_memory;
    long_t const bias_memory   ;
    long_t const batch_size    ;

    conv_layer( long_t n, long_t fin, long_t fout,
                vec3i const & is, vec3i const & ks ) noexcept
        : Base(n, fin, is, n, fout, is + vec3i::one - ks)
        , kernel_size(ks)
        , kernel_len(ks[0]*ks[1]*ks[2])
        , kernels_len(kernel_len*fin*fout)
        , kernels_memory(kernels_len*sizeof(float))
        , bias_memory(fout*sizeof(float))
        , batch_size(n)
    { }
};

template<typename Base>
class pool_layer: public Base
{
public:
    vec3i const window_size;

    pool_layer( long_t n, long_t finout,
                vec3i const & is, vec3i const & ws ) noexcept
        : Base(n, finout, is, n, finout, is / ws)
        , window_size(ws)
    {
        STRONG_ASSERT( is % ws == vec3i::zero );
    }
};

template<typename Base>
class mfp_layer: public Base
{
public:
    vec3i  const window_size  ;
    long_t const num_fragments;

    mfp_layer( long_t n, long_t finout,
               vec3i const & is, vec3i const & ws ) noexcept
        : Base(n, finout, is, n*ws[0]*ws[1]*ws[2], finout, is / ws)
        , window_size(ws)
        , num_fragments(ws[0]*ws[1]*ws[2])
    {
        STRONG_ASSERT( (is+vec3i::one) % ws == vec3i::zero );
    }
};


}} // namespace znn::fwd
