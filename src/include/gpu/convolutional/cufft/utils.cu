#include "utils.hpp"

#include "../../utils.hpp"

#include <thrust/transform.h>
#include <thrust/complex.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace znn { namespace fwd {

void div_all_by( float* first, float* last, float val ) noexcept
{
    thrust::transform(thrust::device,
                      first, last, first,
                      thrust::placeholders::_1 / val);
}

void add_to( cuComplex* first, cuComplex* last,
             cuComplex* out, float beta) noexcept
{
    float* f = reinterpret_cast<float*>(first);
    float* l = reinterpret_cast<float*>(last);
    float* o = reinterpret_cast<float*>(out);

    thrust::transform
        (thrust::device, f, l, o, o,
         thrust::placeholders::_1 + beta * thrust::placeholders::_2 );
}

void mul_add( cuComplex* first1, cuComplex* last1,
              cuComplex* first2, cuComplex* result ) noexcept
{
    typedef thrust::complex<float> cplx;

    thrust::multiplies<cplx> op;

    cplx* cfirst1 = reinterpret_cast<cplx*>(first1);
    cplx* clast1  = reinterpret_cast<cplx*>(last1);
    cplx* cfirst2 = reinterpret_cast<cplx*>(first2);
    cplx* cresult = reinterpret_cast<cplx*>(result);

    thrust::transform(thrust::device,
                      cfirst1, clast1, cfirst2, cresult, op);

}

struct implode_functor: public thrust::unary_function<int, int>
{
    int xi, yi, xo, yo, xa, ya, za;

    __host__ __device__
    implode_functor(vec3i const & i, vec3i const & o, vec3i const & a)
        : xi(i[2]*i[1]), yi(i[2])
        , xo(o[2]*o[1]), yo(o[2])
        , xa(a[0]), ya(a[1]), za(a[2])
    {}

    __host__ __device__
    int operator()(int a)
    {
        int r = (a/xi + xa) * xo;
        a %= xi;
        r += (a/yi + ya) * yo;
        return r + (a % yi) + za;
    }
};


struct explode_functor: public thrust::unary_function<int, int>
{
    int xi, yi, zi, xo, yo, zo;

    __host__ __device__
    explode_functor(vec3i const & i, vec3i const & o)
        : xi(i[2]*i[1]*i[0]), yi(i[2]*i[1]), zi(i[2])
        , xo(o[2]*o[1]*o[0]), yo(o[2]*o[1]), zo(o[2])
    {}

    __host__ __device__
    int operator()(int a)
    {
        int r = (a/xi) * xo;
        a %= xi;
        r += (a/yi) * yo;
        a %= yi;
        r += (a/zi) * zo;

       return r + (a % zi);
    }
};


kernel_exploder::kernel_exploder( int* w,
                                  vec3i const & k,
                                  vec3i const & e,
                                  size_t n)
    : workspace(w)
    , len(k[0]*k[1]*k[2]*n)
    , olen(e[0]*e[1]*e[2]*n)
{
    explode_functor f(k,e);
    thrust::sequence(thrust::device, w, w + k[0]*k[1]*k[2]*n);
    thrust::transform(thrust::device, w, w + k[0]*k[1]*k[2]*n, w, f);
}

void kernel_exploder::explode( float* in, float* out )
{
    checkCudaErrors( cudaMemset( out, 0, olen * sizeof(float) ));
    thrust::scatter(thrust::device, in, in + len, workspace, out);
}


image_imploder::image_imploder( int* w,
                                vec3i const & is,
                                vec3i const & fs )
    : workspace(w)
{
    vec3i os = is - fs + vec3i::one;

    len = os[0] * os[1] * os[2];

    vec3i off = fs - vec3i::one;

    implode_functor f(os, is, off);
    thrust::sequence(thrust::device, w, w + len);
    thrust::transform(thrust::device, w, w + len, w, f);
}

void image_imploder::implode( float* in, float* out )
{
    thrust::gather(thrust::device, workspace, workspace + len, in, out);
}


struct image_scatter_helper_functor: public thrust::unary_function<int, int>
{
    int xi, yi, xo, yo;

    __host__ __device__
    image_scatter_helper_functor(vec3i const & i, vec3i const & o)
        : xi(i[2]*i[1]), yi(i[2])
        , xo(o[2]*o[1]), yo(o[2])
    {}

    __host__ __device__
    int operator()(int a)
    {
        int r = ( a / xi ) * xo;
        a %= xi;
        r += ( a / yi ) * yo;
        return r + ( a % yi );
    }
};


image_scatter::image_scatter( int * w, vec3i const & is, vec3i const & os)
    : workspace(w)
    , len(is[0]*is[1]*is[2])
    , olen(os[0]*os[1]*os[2])
{
    image_scatter_helper_functor f(is,os);
    thrust::sequence(thrust::device, w, w + len);
    thrust::transform(thrust::device, w, w + len, w, f);
}

void image_scatter::scatter( float* in, float* out )
{
    checkCudaErrors( cudaMemset( out, 0, olen * sizeof(float) ));
    thrust::scatter(thrust::device, in, in + len, workspace, out);
}

image_gather::image_gather( int * w,
                            vec3i const & is,
                            vec3i const & os,
                            vec3i const & off)
    : workspace(w)
    , len(is[0]*is[1]*is[2])
{
    implode_functor f(is, os, off);
    thrust::sequence(thrust::device, w, w + len);
    thrust::transform(thrust::device, w, w + len, w, f);
}

void image_gather::gather( float* in, float* out )
{
    thrust::gather(thrust::device, workspace, workspace + len, in, out);
}


/////////////////////////////////////////
//
// This is for doing 1D FFTs

template <typename T>
struct stage_1_functor : public thrust::unary_function<T,T>
{
    T i_l, o_l;

    __host__ __device__
    stage_1_functor(T a, T b): i_l(a), o_l(b) {}

    __host__ __device__
    T operator()(T i) const
    {
        return (i / i_l) * o_l + (i % i_l);
    }
};


//           i_x                   o_x
//     /------------\       /----------------\
//     |123..       |       |1ax             |
// i_y |abc..       |       |2by             |
//     |xyz..       |   i_x |3cz
//     \------------/       |...
//                          |...
//                          .

template<typename T>
struct stage_2_functor : public thrust::unary_function<T,T>
{
    T i_x, i_y, o_x, i_s, o_s;

    __host__ __device__
    stage_2_functor( T ix, T iy, T ox )
        : i_x(ix), i_y(iy), o_x(ox), i_s(ix*iy), o_s(ix*ox) {}

    __host__ __device__
    T operator()(T i) const
    {
        return
            ((i / i_s) * o_s) +
            ((i % i_x) * o_x) +
            ((i / i_x) % i_y);
    }
};


// Find log2(x) and optionally round up to the next integer logarithm.
int find_log2( int x, bool round_up = false )
{
    int a = 31 - __builtin_clz(x);
    if (round_up) a += !(0 == (x & (x - 1)));
    return a;
}

__host__ __device__  __forceinline__ uint umulhi(uint x, uint y)
{
#if __CUDA_ARCH__ >= 100
    return __umulhi(x, y);
#else
    uint64_t product = static_cast<uint64_t>(x) * y;
    return static_cast<uint>(product>> 32);
#endif
}


// Fast division for 31-bit integers.
// Uses the method in Hacker's Delight (2nd edition) page 228.
// Evaluates for denom > 1 and x < 2^31.
struct fast_divide
{
    uint coef;
    uint shift;

    __host__ __device__ __forceinline__ uint divide(uint x) const
    {
        return umulhi(x, coef)>> shift;
    }
    explicit fast_divide( uint d )
    {
        uint p = 31 + find_log2(d, true);
        coef = static_cast<uint>(((1ull<< p) + d - 1) / d);
        shift = p - 32;
    }
};

/////////////////////////////////////////
//
// This is for doing 1D FFTs

struct fast_stage_1_functor : public thrust::unary_function<uint,uint>
{
    uint i_l, o_l;
    fast_divide div_i_l;

    fast_stage_1_functor(uint a, uint b): i_l(a), o_l(b), div_i_l(a) {}

    __host__ __device__ __forceinline__
    uint operator()(uint i) const
    {
        uint r = div_i_l.divide(i);   //  r = i / i_l;
        i -= r * i_l;                 //  i = i % i_l;
        return r * o_l + i;
    }
};

struct fast_stage_1_functor_1 : public thrust::unary_function<uint,uint>
{
    uint o_l;

    fast_stage_1_functor_1(uint b): o_l(b) {}

    __host__ __device__ __forceinline__
    uint operator()(uint i) const
    {
        return i * o_l;
    }
};


//           i_x                   o_x
//     /------------\       /----------------\
//     |123..       |       |1ax             |
// i_y |abc..       |       |2by             |
//     |xyz..       |   i_x |3cz
//     \------------/       |...
//                          |...
//                          .

struct fast_stage_2_functor : public thrust::unary_function<uint,uint>
{
    uint i_x, i_y, o_x;
    fast_divide div_i_x;
    fast_divide div_i_y;

    fast_stage_2_functor( uint ix, uint iy, uint ox )
        : i_x(ix), i_y(iy), o_x(ox), div_i_x(ix), div_i_y(iy) {}

    __host__ __device__ __forceinline__
    uint operator()(uint i) const
    {
        uint b = div_i_x.divide(i);   // b = i / i_x;
        i -= b * i_x;                 // i = i % i_x;

        uint r = div_i_y.divide(b);   // r = (i / i_x) / i_y;
        b -= r * i_y;                 // b = (i / i_x) % i_y;

        return ( r * i_x + i ) * o_x + b;
    }
};


struct fast_stage_2_functor_1_N : public thrust::unary_function<uint,uint>
{
    uint i_y, o_x;
    fast_divide div_i_y;

    fast_stage_2_functor_1_N( uint iy, uint ox )
        : i_y(iy), o_x(ox), div_i_y(iy) {}

    __host__ __device__ __forceinline__
    uint operator()(uint i) const
    {
        uint r = div_i_y.divide(i);
        i -= r * i_y;

        return r * o_x + i;
    }
};


struct fast_stage_2_functor_N_1 : public thrust::unary_function<uint,uint>
{
    uint i_x, o_x;
    fast_divide div_i_x;

    fast_stage_2_functor_N_1( uint ix, uint ox )
        : i_x(ix), o_x(ox), div_i_x(ix) {}

    __host__ __device__ __forceinline__
    uint operator()(uint i) const
    {
        uint b = div_i_x.divide(i);   // b = i / i_x;
        i -= b * i_x;                 // i = i % i_x;

        return ( b * i_x + i ) * o_x;
    }
};

struct fast_stage_2_functor_1_1 : public thrust::unary_function<uint,uint>
{
    uint o_x;

    fast_stage_2_functor_1_1( uint ox )
        : o_x(ox) {}

    __host__ __device__ __forceinline__
    uint operator()(uint i) const
    {
        return i * o_x;
    }
};


void stage_2_scatter( int i_x, int i_y, int o_x,
                      cuComplex* in, cuComplex* out, long_t n ) noexcept
{
    checkCudaErrors( cudaMemset( out, 0, (n/i_y) * o_x * sizeof(cuComplex) ));

    if ( i_x == 1 )
    {
        if ( i_y == 1 )
        {
            thrust::scatter(
                thrust::device,
                in, in + n,
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor_1_1(o_x)), out);
        }
        else
        {
            thrust::scatter(
                thrust::device,
                in, in + n,
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor_1_N(i_y,o_x)), out);
        }
    }
    else
    {
        if ( i_y == 1 )
        {
            thrust::scatter(
                thrust::device,
                in, in + n,
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor_N_1(i_x,o_x)), out);
        }
        else
        {
            thrust::scatter(
                thrust::device,
                in, in + n,
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor(i_x,i_y,o_x)), out);
        }
    }
}


void stage_2_gather( int i_x, int i_y, int o_x,
                     cuComplex* in, cuComplex* out, long_t n ) noexcept
{
    if ( i_x == 1 )
    {
        if ( i_y == 1 )
        {
            thrust::gather(
                thrust::device,
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor_1_1(o_x)),
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor_1_1(o_x)) + n,
                out, in);
        }
        else
        {
            thrust::gather(
                thrust::device,
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor_1_N(i_y,o_x)),
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor_1_N(i_y,o_x)) + n,
                out, in);
        }
    }
    else
    {
        if ( i_y == 1 )
        {
            thrust::gather(
                thrust::device,
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor_N_1(i_x,o_x)),
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor_N_1(i_x,o_x)) + n,
                out, in);
        }
        else
        {
            thrust::gather(
                thrust::device,
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor(i_x,i_y,o_x)),
                thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                                fast_stage_2_functor(i_x,i_y,o_x)) + n,
                out, in);
        }
    }

}


void stage_1_scatter( int i_x, int o_x,
                      float* in, float* out, long_t n ) noexcept
{
    checkCudaErrors( cudaMemset( out, 0, (n/i_x) * o_x * sizeof(float) ));

    if ( i_x == 1 )
    {
        thrust::scatter(
            thrust::device,
            in, in + n,
            thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                            fast_stage_1_functor_1(o_x)), out);
    }
    else
    {
        thrust::scatter(
            thrust::device,
            in, in + n,
            thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                            fast_stage_1_functor(i_x,o_x)), out);
    }
}



void stage_1_gather( int i_x, int o_x,
                     float* in, float* out, long_t n ) noexcept
{
    if ( i_x == 1 )
    {
        thrust::gather(
            thrust::device,
            thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                            fast_stage_1_functor(i_x,o_x)),
            thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                            fast_stage_1_functor(i_x,o_x))+n,
            out, in);
    }
    else
    {
        thrust::gather(
            thrust::device,
            thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                            fast_stage_1_functor_1(o_x)),
            thrust::make_transform_iterator(thrust::counting_iterator<uint>(0),
                                            fast_stage_1_functor_1(o_x))+n,
            out, in);
    }
}


}} // namespace znn::fwd
