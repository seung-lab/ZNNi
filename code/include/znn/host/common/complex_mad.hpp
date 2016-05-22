#include <x86intrin.h>
#include <immintrin.h>

#include <cstddef>
#include <complex>

namespace znn { namespace fwd { namespace host {

inline void complex_mad_0( float const *, float const *, float * ) noexcept
{}

inline void complex_mad_1( float const * a, float const * b, float * c ) noexcept
{
    __m128 xmm0;
    __m128 xmm1;
    __m128 xmm2;
    __m128 xmm3;
    __m128 xmm4;

    xmm0 = _mm_load_ps(a);
    xmm1 = _mm_load_ps(b);
    xmm4 = _mm_load_ps(c);

    xmm3 = _mm_shuffle_ps(xmm1,xmm1,0xB1);   // Swap b.re and b.im
    xmm2 = _mm_shuffle_ps(xmm0,xmm0,0xF5);   // Imag. part of a in both
    xmm0 = _mm_shuffle_ps(xmm0,xmm0,0xA0);   // Real part of a in both
    xmm2 = _mm_mul_ps(xmm2, xmm3);           // (a.im*b.im, a.im*b.re)

#ifdef __FMA__      // FMA3
    xmm0 =  _mm_fmaddsub_ps(xmm0, xmm1, xmm2);      // a_re * b +/- aib
#elif defined (__FMA4__)  // FMA4
    xmm0 =  _mm_maddsub_ps(xmm0, xmm1, xmm2);       // a_re * b +/- aib
#else
    xmm0 = _mm_mul_ps(xmm0, xmm1);        // (a.re*b.re, a.re*b.im)
    xmm0 = _mm_addsub_ps(xmm0, xmm2);     // subtract/add
#endif

    xmm0 = _mm_add_ps(xmm0,xmm4);

    _mm_store_sd((double*)c, _mm_castps_pd(xmm0));
}

inline void complex_mad_2( float const * a, float const * b, float* c ) noexcept
{
    __m128 xmm0;
    __m128 xmm1;
    __m128 xmm2;
    __m128 xmm3;
    __m128 xmm4;

    xmm0 = _mm_load_ps(a);
    xmm1 = _mm_load_ps(b);
    xmm4 = _mm_load_ps(c);

    xmm3 = _mm_shuffle_ps(xmm1,xmm1,0xB1);   // Swap b.re and b.im
    xmm2 = _mm_shuffle_ps(xmm0,xmm0,0xF5);   // Imag. part of a in both
    xmm0 = _mm_shuffle_ps(xmm0,xmm0,0xA0);   // Real part of a in both
    xmm2 = _mm_mul_ps(xmm2, xmm3);           // (a.im*b.im, a.im*b.re)

#ifdef __FMA__      // FMA3
    xmm0 =  _mm_fmaddsub_ps(xmm0, xmm1, xmm2);      // a_re * b +/- aib
#elif defined (__FMA4__)  // FMA4
    xmm0 =  _mm_maddsub_ps(xmm0, xmm1, xmm2);       // a_re * b +/- aib
#else
    xmm0 = _mm_mul_ps(xmm0, xmm1);        // (a.re*b.re, a.re*b.im)
    xmm0 = _mm_addsub_ps(xmm0, xmm2);     // subtract/add
#endif

    xmm0 = _mm_add_ps(xmm0,xmm4);

    _mm_store_ps(c, xmm0);
}

inline void complex_mad_3( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_2(a,b,c);
    complex_mad_1(a+4,b+4,c+4);
}

inline void complex_mad_4( float const * a, float const * b, float* c ) noexcept
{
    __m256 ymm0;
    __m256 ymm1;
    __m256 ymm2;
    __m256 ymm3;
    __m256 ymm4;

    ymm0 = _mm256_load_ps(a);
    ymm1 = _mm256_load_ps(b);
    ymm4 = _mm256_load_ps(c);

    ymm3 = _mm256_shuffle_ps(ymm1,ymm1,0xB1);
    ymm2 = _mm256_shuffle_ps(ymm0,ymm0,0xF5);
    ymm0 = _mm256_shuffle_ps(ymm0,ymm0,0xA0);
    ymm2 = _mm256_mul_ps(ymm2, ymm3);           // aib

#ifdef __FMA__      // FMA3
    ymm0 =  _mm256_fmaddsub_ps(ymm0, ymm1, ymm2);      // a_re * b +/- aib
#elif defined (__FMA4__)  // FMA4
    ymm0 =  _mm256_maddsub_ps(ymm0, ymm1, ymm2);      // a_re * b +/- aib
#else
    ymm0 = _mm256_mul_ps(ymm0, ymm1);        // (a.re*b.re, a.re*b.im)
    ymm0 = _mm256_addsub_ps(ymm0, ymm2);    // subtract/add
#endif  // FMA

    ymm0 = _mm256_add_ps(ymm0,ymm4);
    _mm256_store_ps(c, ymm0);
}

inline void complex_mad_5( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_4(a,b,c);
    complex_mad_1(a+8,b+8,c+8);
}

inline void complex_mad_6( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_4(a,b,c);
    complex_mad_2(a+8,b+8,c+8);
}

inline void complex_mad_7( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_4(a,b,c);
    complex_mad_3(a+8,b+8,c+8);
}

inline void complex_mad_8( float const * a, float const * b, float* c ) noexcept
{
    __m256 ymm0a;
    __m256 ymm1a;
    __m256 ymm2a;
    __m256 ymm3a;
    __m256 ymm4a;

    __m256 ymm0b;
    __m256 ymm1b;
    __m256 ymm2b;
    __m256 ymm3b;
    __m256 ymm4b;

    ymm0a = _mm256_load_ps(a);
    ymm0b = _mm256_load_ps(a+8);

    ymm1a = _mm256_load_ps(b);
    ymm1b = _mm256_load_ps(b+8);

    ymm4a = _mm256_load_ps(c);
    ymm4b = _mm256_load_ps(c+8);

    ymm3a = _mm256_shuffle_ps(ymm1a,ymm1a,0xB1);
    ymm3b = _mm256_shuffle_ps(ymm1b,ymm1b,0xB1);

    ymm2a = _mm256_shuffle_ps(ymm0a,ymm0a,0xF5);
    ymm2b = _mm256_shuffle_ps(ymm0b,ymm0b,0xF5);

    ymm0a = _mm256_shuffle_ps(ymm0a,ymm0a,0xA0);
    ymm0b = _mm256_shuffle_ps(ymm0b,ymm0b,0xA0);

    ymm2a = _mm256_mul_ps(ymm2a, ymm3a);           // aib
    ymm2b = _mm256_mul_ps(ymm2b, ymm3b);           // aib

#ifdef __FMA__      // FMA3
    ymm0a =  _mm256_fmaddsub_ps(ymm0a, ymm1a, ymm2a);      // a_re * b +/- aib
    ymm0b =  _mm256_fmaddsub_ps(ymm0b, ymm1b, ymm2b);      // a_re * b +/- aib
#elif defined (__FMA4__)  // FMA4
    ymm0a =  _mm256_maddsub_ps(ymm0a, ymm1a, ymm2a);      // a_re * b +/- aib
    ymm0b =  _mm256_maddsub_ps(ymm0b, ymm1b, ymm2b);      // a_re * b +/- aib
#else
    ymm0a = _mm256_mul_ps(ymm0a, ymm1a);        // (a.re*b.re, a.re*b.im)
    ymm0b = _mm256_mul_ps(ymm0b, ymm1b);        // (a.re*b.re, a.re*b.im)

    ymm0a = _mm256_addsub_ps(ymm0a, ymm2a);    // subtract/add
    ymm0b = _mm256_addsub_ps(ymm0b, ymm2b);    // subtract/add
#endif  // FMA

    ymm0a = _mm256_add_ps(ymm0a,ymm4a);
    ymm0b = _mm256_add_ps(ymm0b,ymm4b);

    _mm256_store_ps(c, ymm0a);
    _mm256_store_ps(c+8, ymm0b);
}

inline void complex_mad_9( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_8(a,b,c);
    complex_mad_1(a+16,b+16,c+16);
}

inline void complex_mad_10( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_8(a,b,c);
    complex_mad_2(a+16,b+16,c+16);
}

inline void complex_mad_11( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_8(a,b,c);
    complex_mad_3(a+16,b+16,c+16);
}

inline void complex_mad_12( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_8(a,b,c);
    complex_mad_4(a+16,b+16,c+16);
}

inline void complex_mad_13( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_8(a,b,c);
    complex_mad_5(a+16,b+16,c+16);
}

inline void complex_mad_14( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_8(a,b,c);
    complex_mad_6(a+16,b+16,c+16);
}

inline void complex_mad_15( float const * a, float const * b, float* c ) noexcept
{
    complex_mad_8(a,b,c);
    complex_mad_7(a+16,b+16,c+16);
}

inline void complex_mad_16( float const * a, float const * b, float* c ) noexcept
{
    __m256 ymm0x;
    __m256 ymm1x;
    __m256 ymm2x;
    __m256 ymm3x;

    __m256 ymm0a;
    __m256 ymm1a;
    __m256 ymm2a;
    __m256 ymm3a;

    __m256 ymm0b;
    __m256 ymm1b;
    __m256 ymm2b;
    __m256 ymm3b;

    __m256 ymm0c;
    __m256 ymm1c;
    __m256 ymm2c;
    __m256 ymm3c;

    ymm0x = _mm256_load_ps(a);
    ymm1x = _mm256_load_ps(b);

    ymm0a = _mm256_load_ps(a+8);
    ymm1a = _mm256_load_ps(b+8);

    ymm0b = _mm256_load_ps(a+16);
    ymm1b = _mm256_load_ps(b+16);

    ymm0c = _mm256_load_ps(a+24);
    ymm1c = _mm256_load_ps(b+24);

    ymm3x = _mm256_shuffle_ps(ymm1x,ymm1x,0xB1);   // Swap b.re and b.im     b_flip
    ymm3a = _mm256_shuffle_ps(ymm1a,ymm1a,0xB1);   // Swap b.re and b.im     b_flip
    ymm3b = _mm256_shuffle_ps(ymm1b,ymm1b,0xB1);   // Swap b.re and b.im     b_flip
    ymm3c = _mm256_shuffle_ps(ymm1c,ymm1c,0xB1);   // Swap b.re and b.im     b_flip

    ymm2x = _mm256_shuffle_ps(ymm0x,ymm0x,0xF5);   // Imag part of a in both  a_im
    ymm2a = _mm256_shuffle_ps(ymm0a,ymm0a,0xF5);   // Imag part of a in both  a_im
    ymm2b = _mm256_shuffle_ps(ymm0b,ymm0b,0xF5);   // Imag part of a in both  a_im
    ymm2c = _mm256_shuffle_ps(ymm0c,ymm0c,0xF5);   // Imag part of a in both  a_im

    ymm0x = _mm256_shuffle_ps(ymm0x,ymm0x,0xA0);   // Real part of a in both a_re
    ymm0a = _mm256_shuffle_ps(ymm0a,ymm0a,0xA0);   // Real part of a in both a_re
    ymm0b = _mm256_shuffle_ps(ymm0b,ymm0b,0xA0);   // Real part of a in both a_re
    ymm0c = _mm256_shuffle_ps(ymm0c,ymm0c,0xA0);   // Real part of a in both a_re

    ymm2x = _mm256_mul_ps(ymm2x, ymm3x);           // aib
    ymm2a = _mm256_mul_ps(ymm2a, ymm3a);           // aib
    ymm2b = _mm256_mul_ps(ymm2b, ymm3b);           // aib
    ymm2c = _mm256_mul_ps(ymm2c, ymm3c);           // aib

    ymm3x = _mm256_load_ps(c);
    ymm3a = _mm256_load_ps(c+8);
    ymm3b = _mm256_load_ps(c+16);
    ymm3c = _mm256_load_ps(c+24);

#ifdef __FMA__      // FMA3

    ymm0x =  _mm256_fmaddsub_ps(ymm0x, ymm1x, ymm2x);      // a_re * b +/- aib
    ymm0a =  _mm256_fmaddsub_ps(ymm0a, ymm1a, ymm2a);      // a_re * b +/- aib
    ymm0b =  _mm256_fmaddsub_ps(ymm0b, ymm1b, ymm2b);      // a_re * b +/- aib
    ymm0c =  _mm256_fmaddsub_ps(ymm0c, ymm1c, ymm2c);      // a_re * b +/- aib

#elif defined (__FMA4__)  // FMA4

    ymm0x =  _mm256_maddsub_ps(ymm0x, ymm1x, ymm2x);      // a_re * b +/- aib
    ymm0a =  _mm256_maddsub_ps(ymm0a, ymm1a, ymm2a);      // a_re * b +/- aib
    ymm0b =  _mm256_maddsub_ps(ymm0b, ymm1b, ymm2b);      // a_re * b +/- aib
    ymm0c =  _mm256_maddsub_ps(ymm0c, ymm1c, ymm2c);      // a_re * b +/- aib

#else

    ymm0x = _mm256_mul_ps(ymm0x, ymm1x);        // (a.re*b.re, a.re*b.im)
    ymm0a = _mm256_mul_ps(ymm0a, ymm1a);        // (a.re*b.re, a.re*b.im)
    ymm0b = _mm256_mul_ps(ymm0b, ymm1b);        // (a.re*b.re, a.re*b.im)
    ymm0c = _mm256_mul_ps(ymm0c, ymm1c);        // (a.re*b.re, a.re*b.im)

    ymm0x = _mm256_addsub_ps(ymm0x, ymm2x);    // subtract/add
    ymm0a = _mm256_addsub_ps(ymm0a, ymm2a);    // subtract/add
    ymm0b = _mm256_addsub_ps(ymm0b, ymm2b);    // subtract/add
    ymm0c = _mm256_addsub_ps(ymm0c, ymm2c);    // subtract/add

#endif  // FMA

    ymm0x = _mm256_add_ps(ymm0x,ymm3x);
    ymm0a = _mm256_add_ps(ymm0a,ymm3a);
    ymm0b = _mm256_add_ps(ymm0b,ymm3b);
    ymm0c = _mm256_add_ps(ymm0c,ymm3c);

    _mm256_store_ps(c   , ymm0x);
    _mm256_store_ps(c+8 , ymm0a);
    _mm256_store_ps(c+16, ymm0b);
    _mm256_store_ps(c+24, ymm0c);

}

inline void complex_mad( float const * a, float const * b, float * c, std::size_t n ) noexcept
{
    typedef decltype(&complex_mad_1) fn_type;
    static fn_type dispatch[16] =
        { complex_mad_0, complex_mad_1, complex_mad_2, complex_mad_3,
          complex_mad_4, complex_mad_5, complex_mad_6, complex_mad_7,
          complex_mad_8, complex_mad_9, complex_mad_10, complex_mad_11,
          complex_mad_12, complex_mad_13, complex_mad_14, complex_mad_15 };

    for ( std::size_t i = 0; i < n / 16; ++i )
    {
        complex_mad_16(a,b,c);
        a += 32; b += 32; c += 32;
    }

    (dispatch[n%16])(a,b,c);
}


inline void complex_mad( std::complex<float> const * a,
                         std::complex<float> const * b,
                         std::complex<float> * c,
                         std::size_t n ) noexcept
{
    complex_mad(reinterpret_cast<float const *>(a),
                reinterpret_cast<float const *>(b),
                reinterpret_cast<float*>(c), n);
}


inline void complex_mst_0( float const *, float const *, float * ) noexcept
{}

inline void complex_mst_1( float const * a, float const * b, float * c ) noexcept
{
    __m128 xmm0;
    __m128 xmm1;
    __m128 xmm2;
    __m128 xmm3;

    xmm0 = _mm_load_ps(a);
    xmm1 = _mm_load_ps(b);

    xmm3 = _mm_shuffle_ps(xmm1,xmm1,0xB1);   // Swap b.re and b.im
    xmm2 = _mm_shuffle_ps(xmm0,xmm0,0xF5);   // Imag. part of a in both
    xmm0 = _mm_shuffle_ps(xmm0,xmm0,0xA0);   // Real part of a in both
    xmm2 = _mm_mul_ps(xmm2, xmm3);           // (a.im*b.im, a.im*b.re)

#ifdef __FMA__      // FMA3
    xmm0 =  _mm_fmaddsub_ps(xmm0, xmm1, xmm2);      // a_re * b +/- aib
#elif defined (__FMA4__)  // FMA4
    xmm0 =  _mm_maddsub_ps(xmm0, xmm1, xmm2);       // a_re * b +/- aib
#else
    xmm0 = _mm_mul_ps(xmm0, xmm1);        // (a.re*b.re, a.re*b.im)
    xmm0 = _mm_addsub_ps(xmm0, xmm2);     // subtract/add
#endif

    _mm_store_sd((double*)c, _mm_castps_pd(xmm0));
}

inline void complex_mst_2( float const * a, float const * b, float* c ) noexcept
{
    __m128 xmm0;
    __m128 xmm1;
    __m128 xmm2;
    __m128 xmm3;

    xmm0 = _mm_load_ps(a);
    xmm1 = _mm_load_ps(b);

    xmm3 = _mm_shuffle_ps(xmm1,xmm1,0xB1);   // Swap b.re and b.im
    xmm2 = _mm_shuffle_ps(xmm0,xmm0,0xF5);   // Imag. part of a in both
    xmm0 = _mm_shuffle_ps(xmm0,xmm0,0xA0);   // Real part of a in both
    xmm2 = _mm_mul_ps(xmm2, xmm3);           // (a.im*b.im, a.im*b.re)

#ifdef __FMA__      // FMA3
    xmm0 =  _mm_fmaddsub_ps(xmm0, xmm1, xmm2);      // a_re * b +/- aib
#elif defined (__FMA4__)  // FMA4
    xmm0 =  _mm_maddsub_ps(xmm0, xmm1, xmm2);       // a_re * b +/- aib
#else
    xmm0 = _mm_mul_ps(xmm0, xmm1);        // (a.re*b.re, a.re*b.im)
    xmm0 = _mm_addsub_ps(xmm0, xmm2);     // subtract/add
#endif

    _mm_store_ps(c, xmm0);
}

inline void complex_mst_3( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_2(a,b,c);
    complex_mst_1(a+4,b+4,c+4);
}

inline void complex_mst_4( float const * a, float const * b, float* c ) noexcept
{
    __m256 ymm0;
    __m256 ymm1;
    __m256 ymm2;
    __m256 ymm3;

    ymm0 = _mm256_load_ps(a);
    ymm1 = _mm256_load_ps(b);

    ymm3 = _mm256_shuffle_ps(ymm1,ymm1,0xB1);
    ymm2 = _mm256_shuffle_ps(ymm0,ymm0,0xF5);
    ymm0 = _mm256_shuffle_ps(ymm0,ymm0,0xA0);
    ymm2 = _mm256_mul_ps(ymm2, ymm3);           // aib

#ifdef __FMA__      // FMA3
    ymm0 =  _mm256_fmaddsub_ps(ymm0, ymm1, ymm2);      // a_re * b +/- aib
#elif defined (__FMA4__)  // FMA4
    ymm0 =  _mm256_maddsub_ps(ymm0, ymm1, ymm2);      // a_re * b +/- aib
#else
    ymm0 = _mm256_mul_ps(ymm0, ymm1);        // (a.re*b.re, a.re*b.im)
    ymm0 = _mm256_addsub_ps(ymm0, ymm2);    // subtract/add
#endif  // FMA

    _mm256_store_ps(c, ymm0);
}

inline void complex_mst_5( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_4(a,b,c);
    complex_mst_1(a+8,b+8,c+8);
}

inline void complex_mst_6( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_4(a,b,c);
    complex_mst_2(a+8,b+8,c+8);
}

inline void complex_mst_7( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_4(a,b,c);
    complex_mst_3(a+8,b+8,c+8);
}

inline void complex_mst_8( float const * a, float const * b, float* c ) noexcept
{
    __m256 ymm0a;
    __m256 ymm1a;
    __m256 ymm2a;
    __m256 ymm3a;

    __m256 ymm0b;
    __m256 ymm1b;
    __m256 ymm2b;
    __m256 ymm3b;

    ymm0a = _mm256_load_ps(a);
    ymm0b = _mm256_load_ps(a+8);

    ymm1a = _mm256_load_ps(b);
    ymm1b = _mm256_load_ps(b+8);

    ymm3a = _mm256_shuffle_ps(ymm1a,ymm1a,0xB1);
    ymm3b = _mm256_shuffle_ps(ymm1b,ymm1b,0xB1);

    ymm2a = _mm256_shuffle_ps(ymm0a,ymm0a,0xF5);
    ymm2b = _mm256_shuffle_ps(ymm0b,ymm0b,0xF5);

    ymm0a = _mm256_shuffle_ps(ymm0a,ymm0a,0xA0);
    ymm0b = _mm256_shuffle_ps(ymm0b,ymm0b,0xA0);

    ymm2a = _mm256_mul_ps(ymm2a, ymm3a);           // aib
    ymm2b = _mm256_mul_ps(ymm2b, ymm3b);           // aib

#ifdef __FMA__      // FMA3
    ymm0a =  _mm256_fmaddsub_ps(ymm0a, ymm1a, ymm2a);      // a_re * b +/- aib
    ymm0b =  _mm256_fmaddsub_ps(ymm0b, ymm1b, ymm2b);      // a_re * b +/- aib
#elif defined (__FMA4__)  // FMA4
    ymm0a =  _mm256_maddsub_ps(ymm0a, ymm1a, ymm2a);      // a_re * b +/- aib
    ymm0b =  _mm256_maddsub_ps(ymm0b, ymm1b, ymm2b);      // a_re * b +/- aib
#else
    ymm0a = _mm256_mul_ps(ymm0a, ymm1a);        // (a.re*b.re, a.re*b.im)
    ymm0b = _mm256_mul_ps(ymm0b, ymm1b);        // (a.re*b.re, a.re*b.im)

    ymm0a = _mm256_addsub_ps(ymm0a, ymm2a);    // subtract/add
    ymm0b = _mm256_addsub_ps(ymm0b, ymm2b);    // subtract/add
#endif  // FMA

    _mm256_store_ps(c, ymm0a);
    _mm256_store_ps(c+8, ymm0b);
}

inline void complex_mst_9( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_8(a,b,c);
    complex_mst_1(a+16,b+16,c+16);
}

inline void complex_mst_10( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_8(a,b,c);
    complex_mst_2(a+16,b+16,c+16);
}

inline void complex_mst_11( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_8(a,b,c);
    complex_mst_3(a+16,b+16,c+16);
}

inline void complex_mst_12( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_8(a,b,c);
    complex_mst_4(a+16,b+16,c+16);
}

inline void complex_mst_13( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_8(a,b,c);
    complex_mst_5(a+16,b+16,c+16);
}

inline void complex_mst_14( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_8(a,b,c);
    complex_mst_6(a+16,b+16,c+16);
}

inline void complex_mst_15( float const * a, float const * b, float* c ) noexcept
{
    complex_mst_8(a,b,c);
    complex_mst_7(a+16,b+16,c+16);
}

inline void complex_mst_16( float const * a, float const * b, float* c ) noexcept
{
    __m256 ymm0x;
    __m256 ymm1x;
    __m256 ymm2x;
    __m256 ymm3x;

    __m256 ymm0a;
    __m256 ymm1a;
    __m256 ymm2a;
    __m256 ymm3a;

    __m256 ymm0b;
    __m256 ymm1b;
    __m256 ymm2b;
    __m256 ymm3b;

    __m256 ymm0c;
    __m256 ymm1c;
    __m256 ymm2c;
    __m256 ymm3c;

    ymm0x = _mm256_load_ps(a);
    ymm1x = _mm256_load_ps(b);

    ymm0a = _mm256_load_ps(a+8);
    ymm1a = _mm256_load_ps(b+8);

    ymm0b = _mm256_load_ps(a+16);
    ymm1b = _mm256_load_ps(b+16);

    ymm0c = _mm256_load_ps(a+24);
    ymm1c = _mm256_load_ps(b+24);

    ymm3x = _mm256_shuffle_ps(ymm1x,ymm1x,0xB1);   // Swap b.re and b.im     b_flip
    ymm3a = _mm256_shuffle_ps(ymm1a,ymm1a,0xB1);   // Swap b.re and b.im     b_flip
    ymm3b = _mm256_shuffle_ps(ymm1b,ymm1b,0xB1);   // Swap b.re and b.im     b_flip
    ymm3c = _mm256_shuffle_ps(ymm1c,ymm1c,0xB1);   // Swap b.re and b.im     b_flip

    ymm2x = _mm256_shuffle_ps(ymm0x,ymm0x,0xF5);   // Imag part of a in both  a_im
    ymm2a = _mm256_shuffle_ps(ymm0a,ymm0a,0xF5);   // Imag part of a in both  a_im
    ymm2b = _mm256_shuffle_ps(ymm0b,ymm0b,0xF5);   // Imag part of a in both  a_im
    ymm2c = _mm256_shuffle_ps(ymm0c,ymm0c,0xF5);   // Imag part of a in both  a_im

    ymm0x = _mm256_shuffle_ps(ymm0x,ymm0x,0xA0);   // Real part of a in both a_re
    ymm0a = _mm256_shuffle_ps(ymm0a,ymm0a,0xA0);   // Real part of a in both a_re
    ymm0b = _mm256_shuffle_ps(ymm0b,ymm0b,0xA0);   // Real part of a in both a_re
    ymm0c = _mm256_shuffle_ps(ymm0c,ymm0c,0xA0);   // Real part of a in both a_re

    ymm2x = _mm256_mul_ps(ymm2x, ymm3x);           // aib
    ymm2a = _mm256_mul_ps(ymm2a, ymm3a);           // aib
    ymm2b = _mm256_mul_ps(ymm2b, ymm3b);           // aib
    ymm2c = _mm256_mul_ps(ymm2c, ymm3c);           // aib

#ifdef __FMA__      // FMA3

    ymm0x =  _mm256_fmaddsub_ps(ymm0x, ymm1x, ymm2x);      // a_re * b +/- aib
    ymm0a =  _mm256_fmaddsub_ps(ymm0a, ymm1a, ymm2a);      // a_re * b +/- aib
    ymm0b =  _mm256_fmaddsub_ps(ymm0b, ymm1b, ymm2b);      // a_re * b +/- aib
    ymm0c =  _mm256_fmaddsub_ps(ymm0c, ymm1c, ymm2c);      // a_re * b +/- aib

#elif defined (__FMA4__)  // FMA4

    ymm0x =  _mm256_maddsub_ps(ymm0x, ymm1x, ymm2x);      // a_re * b +/- aib
    ymm0a =  _mm256_maddsub_ps(ymm0a, ymm1a, ymm2a);      // a_re * b +/- aib
    ymm0b =  _mm256_maddsub_ps(ymm0b, ymm1b, ymm2b);      // a_re * b +/- aib
    ymm0c =  _mm256_maddsub_ps(ymm0c, ymm1c, ymm2c);      // a_re * b +/- aib

#else

    ymm0x = _mm256_mul_ps(ymm0x, ymm1x);        // (a.re*b.re, a.re*b.im)
    ymm0a = _mm256_mul_ps(ymm0a, ymm1a);        // (a.re*b.re, a.re*b.im)
    ymm0b = _mm256_mul_ps(ymm0b, ymm1b);        // (a.re*b.re, a.re*b.im)
    ymm0c = _mm256_mul_ps(ymm0c, ymm1c);        // (a.re*b.re, a.re*b.im)

    ymm0x = _mm256_addsub_ps(ymm0x, ymm2x);    // subtract/add
    ymm0a = _mm256_addsub_ps(ymm0a, ymm2a);    // subtract/add
    ymm0b = _mm256_addsub_ps(ymm0b, ymm2b);    // subtract/add
    ymm0c = _mm256_addsub_ps(ymm0c, ymm2c);    // subtract/add

#endif  // FMA

    _mm256_store_ps(c   , ymm0x);
    _mm256_store_ps(c+8 , ymm0a);
    _mm256_store_ps(c+16, ymm0b);
    _mm256_store_ps(c+24, ymm0c);

}

inline void complex_mst( float const * a, float const * b, float * c, std::size_t n ) noexcept
{
    typedef decltype(&complex_mst_1) fn_type;
    static fn_type dispatch[16] =
        { complex_mst_0, complex_mst_1, complex_mst_2, complex_mst_3,
          complex_mst_4, complex_mst_5, complex_mst_6, complex_mst_7,
          complex_mst_8, complex_mst_9, complex_mst_10, complex_mst_11,
          complex_mst_12, complex_mst_13, complex_mst_14, complex_mst_15 };

    for ( std::size_t i = 0; i < n / 16; ++i )
    {
        complex_mst_16(a,b,c);
        a += 32; b += 32; c += 32;
    }

    (dispatch[n%16])(a,b,c);
}


inline void complex_mst( std::complex<float> const * a,
                         std::complex<float> const * b,
                         std::complex<float> * c,
                         std::size_t n ) noexcept
{
    complex_mst(reinterpret_cast<float const *>(a),
                reinterpret_cast<float const *>(b),
                reinterpret_cast<float*>(c), n);
}

}}} // namespace znn::fwd::host
