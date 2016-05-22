#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"

namespace znn { namespace fwd {

class layer2d
{
public:
    long_t in_batch_size   ;
    long_t num_inputs      ;
    vec2i  in_image_size   ;
    long_t in_image_len    ;
    long_t input_len       ;
    long_t total_input_len ;
    long_t input_memory    ;
    vec4i  input_shape     ;

    long_t out_batch_size  ;
    long_t num_outputs     ;
    vec2i  out_image_size  ;
    long_t out_image_len   ;
    long_t output_len      ;
    long_t total_output_len;
    long_t output_memory   ;
    vec4i  output_shape    ;

public:
    layer2d() noexcept {}

    layer2d( long_t nin, long_t fin, vec2i const & is,
             long_t nout, long_t fout, vec2i const & os ) noexcept
        : in_batch_size(nin)
        , num_inputs(fin)
        , in_image_size(is)
        , in_image_len(is[0]*is[1])
        , input_len(fin*in_image_len)
        , total_input_len(input_len*nin)
        , input_memory(total_input_len*sizeof(float))
        , input_shape(nin,fin,is[0],is[1])
        , out_batch_size(nout)
        , num_outputs(fout)
        , out_image_size(os)
        , out_image_len(os[0]*os[1])
        , output_len(fout*out_image_len)
        , total_output_len(output_len*nout)
        , output_memory(total_output_len*sizeof(float))
        , output_shape(nout,fout,os[0],os[1])
    { }

    layer2d& operator=( layer2d const & ) = default;

};

template<typename Base>
class conv_layer2d: public Base
{
public:
    vec2i  kernel_size   ;
    long_t kernel_len    ;
    long_t kernels_len   ;
    long_t kernels_memory;
    long_t bias_memory   ;
    long_t batch_size    ;
    vec4i  kernels_shape ;

    conv_layer2d() noexcept {}

    conv_layer2d( long_t n, long_t fin, long_t fout,
                  vec2i const & is, vec2i const & ks ) noexcept
        : Base(n, fin, is, n, fout, is + vec2i::one - ks)
        , kernel_size(ks)
        , kernel_len(ks[0]*ks[1])
        , kernels_len(kernel_len*fin*fout)
        , kernels_memory(kernels_len*sizeof(float))
        , bias_memory(fout*sizeof(float))
        , batch_size(n)
        , kernels_shape(fout,fin,ks[0],ks[1])
    { }

    conv_layer2d& operator=( conv_layer2d const & ) = default;

};

template<typename Base>
class pool_layer2d: public Base
{
public:
    vec2i window_size;

    pool_layer2d() noexcept {}

    pool_layer2d( long_t n, long_t finout,
                  vec2i const & is, vec2i const & ws ) noexcept
        : Base(n, finout, is, n, finout, is / ws)
        , window_size(ws)
    {
        STRONG_ASSERT( is % ws == vec2i::zero );
    }

    pool_layer2d& operator=( pool_layer2d const & ) = default;
};

template<typename Base>
class softmax_layer2d: public Base
{
public:
    softmax_layer2d() noexcept {}

    softmax_layer2d( long_t n, long_t finout,
                     vec2i const & is ) noexcept
        : Base(n, finout, is, n, finout, is)
    {
        STRONG_ASSERT( finout > 1 );
    }

    softmax_layer2d& operator=( softmax_layer2d const & ) = default;
};


template<typename Base>
class mfp_layer2d: public Base
{
public:
    vec2i  window_size  ;
    long_t num_fragments;

    mfp_layer2d() noexcept {};

    mfp_layer2d( long_t n, long_t finout,
                 vec2i const & is, vec2i const & ws ) noexcept
        : Base(n, finout, is, n*ws[0]*ws[1], finout, is / ws)
        , window_size(ws)
        , num_fragments(ws[0]*ws[1])
    {
        STRONG_ASSERT( (is+vec2i::one) % ws == vec2i::zero );
    }

    mfp_layer2d & operator=( mfp_layer2d const & ) = default;
};


template<typename Base>
class maxout_layer2d: public Base
{
public:
    long_t factor;

    maxout_layer2d() noexcept {};

    maxout_layer2d( long_t n, long_t fin, long_t fac,
                    vec2i const & is ) noexcept
        : Base(n, fin, is, n, fin/fac, is)
        , factor(fac)
    {
        STRONG_ASSERT(fac>1);
        STRONG_ASSERT((fin%fac)==0);
    }

    maxout_layer2d & operator=( maxout_layer2d const & ) = default;
};



template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           conv_layer2d<T> const & p)
{
    os << "conv_layer2d: "   << p.in_batch_size << ' '
       << p.num_inputs << "->"
       << p.num_outputs << ' '
       << p.in_image_size << ' '
       << p.kernel_size << std::endl;
    return os;
}


template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           pool_layer2d<T> const & p)
{
    os << "pool_layer2d:   " << p.in_batch_size << ' '
       << p.num_outputs << ' '
       << p.in_image_size << ' '
       << p.window_size << std::endl;
    return os;
}

template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           mfp_layer2d<T> const & p)
{
    os << "mfp_layer2d   : " << p.in_batch_size << "->"
       << p.out_batch_size << ' '
       << p.num_outputs << ' '
       << p.in_image_size << ' '
       << p.window_size << std::endl;
    return os;
}

template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           maxout_layer2d<T> const & p)
{
    os << "maxout_layer2d : " << p.in_batch_size << "->"
       << p.num_inputs << ' '
       << p.num_outputs << ' '
       << p.in_image_size << ' '
       << p.factor << std::endl;
    return os;
}




}} // namespace znn::fwd
