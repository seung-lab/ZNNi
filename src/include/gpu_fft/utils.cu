#include "utils.hpp"

#include "../gpu/cuda_utils.hpp"

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
    float* l = reinterpret_cast<float*>(last) + 1;
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



}} // namespace znn::fwd


using namespace znn::fwd;


int main()
{
    checkCudaErrors( cudaSetDevice(0) );

    float* cx;
    cudaMalloc( &cx, 4*4*4*sizeof(float));

    float* ex;
    cudaMalloc( &ex, 5*5*5*sizeof(float));

    float aca[5*5*5];
    for ( int i = 0; i < 125; ++i ) aca[i] = i;

    for ( int i = 0; i < 5*5*5; ++i )
    {
        if ( i % 5 == 0 ) std::cout << "\n";
        if ( i % 25 == 0 ) std::cout << "\n";
        if ( i % 125 == 0 ) std::cout << "\n";

        std::cout << aca[i] << ' ';
    }


    cudaMemcpy(ex, aca, 5*5*5*sizeof(float), cudaMemcpyHostToDevice);

    int* w;
    cudaMalloc( &w, 5*5*5*sizeof(int));

    image_imploder ke(w, vec3i(5,5,5), vec3i(2,2,2));

    ke.implode(ex, cx);

    float z[4*4*4];

    cudaMemcpy(z, cx, 4*4*4*sizeof(float), cudaMemcpyDeviceToHost);

    // int ww[3*3*3*5];
    // cudaMemcpy(ww, w, 3*3*3*5*sizeof(int), cudaMemcpyDeviceToHost);

    for ( int i = 0; i < 4*4*4; ++i )
    {
        if ( i % 4 == 0 ) std::cout << "\n";
        if ( i % 16 == 0 ) std::cout << "\n";
        if ( i % 64 == 0 ) std::cout << "\n";

        std::cout << z[i] << ' ';
    }

    // for ( int i = 0; i < 3*3*3*5; ++i )
    // {
    //     if ( i % 3 == 0 ) std::cout << "\n";
    //     if ( i % 9 == 0 ) std::cout << "\n";
    //     if ( i % 27 == 0 ) std::cout << "\n";

    //     std::cout << ww[i] << ' ';
    // }


    cudaDeviceReset();
}
