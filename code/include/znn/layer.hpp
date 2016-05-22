#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"

namespace znn { namespace fwd {

class layer
{
public:
    long_t in_batch_size   ;
    long_t num_inputs      ;
    vec3i  in_image_size   ;
    long_t in_image_len    ;
    long_t input_len       ;
    long_t total_input_len ;
    long_t input_memory    ;
    vec5i  input_shape     ;

    long_t out_batch_size  ;
    long_t num_outputs     ;
    vec3i  out_image_size  ;
    long_t out_image_len   ;
    long_t output_len      ;
    long_t total_output_len;
    long_t output_memory   ;
    vec5i  output_shape    ;

public:
    layer() noexcept {}

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

    layer& operator=( layer const & ) = default;

};

template<typename Base>
class conv_layer: public Base
{
public:
    vec3i  kernel_size   ;
    long_t kernel_len    ;
    long_t kernels_len   ;
    long_t kernels_memory;
    long_t bias_memory   ;
    long_t batch_size    ;
    vec5i  kernels_shape ;

    conv_layer() noexcept {}

    conv_layer( long_t n, long_t fin, long_t fout,
                vec3i const & is, vec3i const & ks ) noexcept
        : Base(n, fin, is, n, fout, is + vec3i::one - ks)
        , kernel_size(ks)
        , kernel_len(ks[0]*ks[1]*ks[2])
        , kernels_len(kernel_len*fin*fout)
        , kernels_memory(kernels_len*sizeof(float))
        , bias_memory(fout*sizeof(float))
        , batch_size(n)
        , kernels_shape(fout,fin,ks[0],ks[1],ks[2])
    { }

    conv_layer& operator=( conv_layer const & ) = default;

};

template<typename Base>
class pool_layer: public Base
{
public:
    vec3i window_size;

    pool_layer() noexcept {}

    pool_layer( long_t n, long_t finout,
                vec3i const & is, vec3i const & ws ) noexcept
        : Base(n, finout, is, n, finout, is / ws)
        , window_size(ws)
    {
        STRONG_ASSERT( is % ws == vec3i::zero );
    }

    pool_layer& operator=( pool_layer const & ) = default;
};


template<typename Base>
class maxfilter_layer: public Base
{
public:
    vec3i window_size;

    maxfilter_layer() noexcept {}

    maxfilter_layer( long_t n, long_t finout,
                     vec3i const & is, vec3i const & ws ) noexcept
        : Base(n, finout, is, n, finout, is - ws + vec3i::one)
        , window_size(ws)
    {
        STRONG_ASSERT( is % ws == vec3i::zero );
    }

    maxfilter_layer& operator=( maxfilter_layer const & ) = default;
};



template<typename Base>
class softmax_layer: public Base
{
public:
    softmax_layer() noexcept {}

    softmax_layer( long_t n, long_t finout,
                   vec3i const & is ) noexcept
        : Base(n, finout, is, n, finout, is)
    {
        STRONG_ASSERT( finout > 1 );
    }

    softmax_layer& operator=( softmax_layer const & ) = default;
};



template<typename Base>
class mfp_layer: public Base
{
public:
    vec3i  window_size  ;
    long_t num_fragments;

    mfp_layer() noexcept {};

    mfp_layer( long_t n, long_t finout,
               vec3i const & is, vec3i const & ws ) noexcept
        : Base(n, finout, is, n*ws[0]*ws[1]*ws[2], finout, is / ws)
        , window_size(ws)
        , num_fragments(ws[0]*ws[1]*ws[2])
    {
        STRONG_ASSERT( (is+vec3i::one) % ws == vec3i::zero );
    }

    mfp_layer & operator=( mfp_layer const & ) = default;
};


template<typename Base>
class maxout_layer: public Base
{
public:
    long_t factor;

    maxout_layer() noexcept {};

    maxout_layer( long_t n, long_t fin, long_t fac,
                  vec3i const & is ) noexcept
        : Base(n, fin, is, n, fin/fac, is)
        , factor(fac)
    {
        STRONG_ASSERT(fac>1);
        STRONG_ASSERT((fin%fac)==0);
    }

    maxout_layer & operator=( maxout_layer const & ) = default;
};



template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           conv_layer<T> const & p)
{
    os << "conv_layer: " << p.in_batch_size << ' '
       << p.num_inputs << "->"
       << p.num_outputs << ' '
       << p.in_image_size << ' '
       << p.kernel_size << std::endl;
    return os;
}


template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           pool_layer<T> const & p)
{
    os << "pool_layer: " << p.in_batch_size << ' '
       << p.num_outputs << ' '
       << p.in_image_size << ' '
       << p.window_size << std::endl;
    return os;
}

template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           mfp_layer<T> const & p)
{
    os << "mfp_layer : " << p.in_batch_size << "->"
       << p.out_batch_size << ' '
       << p.num_outputs << ' '
       << p.in_image_size << ' '
       << p.window_size << std::endl;
    return os;
}

template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           maxout_layer<T> const & p)
{
    os << "maxout_layer : " << p.in_batch_size << "->"
       << p.num_inputs << ' '
       << p.num_outputs << ' '
       << p.in_image_size << ' '
       << p.factor << std::endl;
    return os;
}




}} // namespace znn::fwd
