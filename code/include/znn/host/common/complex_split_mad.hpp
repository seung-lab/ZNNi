#include <x86intrin.h>
#include <immintrin.h>

#include <cstddef>
#include <complex>

#ifdef __FMA__
#  define _FNMSUB(a,b,c) _mm256_fnmsub_ps(a,b,c)
#  define _FMADD(a,b,c) _mm256_fmadd_ps(a,b,c)
#else
#  define _FNMSUB(a,b,c) _mm256_sub_ps(c,_mm256_mul_ps(a,b))
#  define _FMADD(a,b,c) _mm256_add_ps(_mm256_mul_ps(a,b),c)
#endif

namespace znn { namespace fwd { namespace host {

inline void complex_split_mst_0( float const *, float const *,
                                 float const *, float const *,
                                 float *, float * ) noexcept
{}

inline void complex_split_mad_0( float const *, float const *,
                                 float const *, float const *,
                                 float *, float * ) noexcept
{}


inline void complex_split_mst_8( float const * ar, float const * ai,
                                 float const * br, float const * bi,
                                 float * rr, float * ri ) noexcept
{
    __m256 ymm0;
    __m256 ymm1;
    __m256 ymm2;
    __m256 ymm3;
    __m256 ymm4;
    __m256 ymm5;

    // rr += ar*br - ai*bi
    // ri += ar*bi + ai*br

    ymm2 = _mm256_load_ps(ar);
    ymm3 = _mm256_load_ps(ai);

    ymm4 = _mm256_load_ps(br);
    ymm5 = _mm256_load_ps(bi);

    ymm0 = _mm256_mul_ps(ymm2,ymm4);
    ymm1 = _mm256_mul_ps(ymm2,ymm5);

    ymm0 = _FNMSUB(ymm3,ymm5,ymm0);
    ymm1 = _FMADD(ymm3,ymm4,ymm1);

    // store

    _mm256_store_ps(rr, ymm0);
    _mm256_store_ps(ri, ymm1);
}

inline void complex_split_mad_8( float const * ar, float const * ai,
                                 float const * br, float const * bi,
                                 float * rr, float * ri ) noexcept
{
    __m256 ymm0;
    __m256 ymm1;
    __m256 ymm2;
    __m256 ymm3;
    __m256 ymm4;
    __m256 ymm5;

    // rr += ar*br - ai*bi
    // ri += ar*bi + ai*br

    ymm0 = _mm256_load_ps(rr);
    ymm1 = _mm256_load_ps(ri);

    ymm2 = _mm256_load_ps(ar);
    ymm3 = _mm256_load_ps(ai);

    ymm4 = _mm256_load_ps(br);
    ymm5 = _mm256_load_ps(bi);

    ymm0 = _FMADD(ymm2,ymm4,ymm0);
    ymm1 = _FNMSUB(ymm2,ymm5,ymm1);

    ymm0 = _FNMSUB(ymm3,ymm5,ymm0);
    ymm1 = _FMADD(ymm3,ymm4,ymm1);

    // store

    _mm256_store_ps(rr, ymm0);
    _mm256_store_ps(ri, ymm1);
}

inline void complex_split_mst_16( float const * ar, float const * ai,
                                  float const * br, float const * bi,
                                  float * rr, float * ri ) noexcept
{
    __m256 ymm0a;
    __m256 ymm1a;
    __m256 ymm2a;
    __m256 ymm3a;
    __m256 ymm4a;
    __m256 ymm5a;

    __m256 ymm0b;
    __m256 ymm1b;
    __m256 ymm2b;
    __m256 ymm3b;
    __m256 ymm4b;
    __m256 ymm5b;


    // rr += ar*br - ai*bi
    // ri += ar*bi + ai*br

    ymm2a = _mm256_load_ps(ar);
    ymm3a = _mm256_load_ps(ai);

    ymm2b = _mm256_load_ps(ar+8);
    ymm3b = _mm256_load_ps(ai+8);

    ymm4a = _mm256_load_ps(br);
    ymm5a = _mm256_load_ps(bi);

    ymm4b = _mm256_load_ps(br+8);
    ymm5b = _mm256_load_ps(bi+8);

    ymm0a = _mm256_mul_ps(ymm2a,ymm4a);
    ymm1a = _mm256_mul_ps(ymm2a,ymm5a);

    ymm0b = _mm256_mul_ps(ymm2b,ymm4b);
    ymm1b = _mm256_mul_ps(ymm2b,ymm5b);

    ymm0a = _FNMSUB(ymm3a,ymm5a,ymm0a);
    ymm1a = _FMADD(ymm3a,ymm4a,ymm1a);

    ymm0b = _FNMSUB(ymm3b,ymm5b,ymm0b);
    ymm1b = _FMADD(ymm3b,ymm4b,ymm1b);

    // store

    _mm256_store_ps(rr, ymm0a);
    _mm256_store_ps(ri, ymm1a);

    _mm256_store_ps(rr+8, ymm0b);
    _mm256_store_ps(ri+8, ymm1b);
}

inline void complex_split_mad_16( float const * ar, float const * ai,
                                  float const * br, float const * bi,
                                  float * rr, float * ri ) noexcept
{
    __m256 ymm0a;
    __m256 ymm1a;
    __m256 ymm2a;
    __m256 ymm3a;
    __m256 ymm4a;
    __m256 ymm5a;

    __m256 ymm0b;
    __m256 ymm1b;
    __m256 ymm2b;
    __m256 ymm3b;
    __m256 ymm4b;
    __m256 ymm5b;


    // rr += ar*br - ai*bi
    // ri += ar*bi + ai*br

    ymm0a = _mm256_load_ps(rr);
    ymm1a = _mm256_load_ps(ri);

    ymm0b = _mm256_load_ps(rr+8);
    ymm1b = _mm256_load_ps(ri+8);

    ymm2a = _mm256_load_ps(ar);
    ymm3a = _mm256_load_ps(ai);

    ymm2b = _mm256_load_ps(ar+8);
    ymm3b = _mm256_load_ps(ai+8);

    ymm4a = _mm256_load_ps(br);
    ymm5a = _mm256_load_ps(bi);

    ymm4b = _mm256_load_ps(br+8);
    ymm5b = _mm256_load_ps(bi+8);

    ymm0a = _FMADD(ymm2a,ymm4a,ymm0a);
    ymm1a = _FNMSUB(ymm2a,ymm5a,ymm1a);

    ymm0b = _FMADD(ymm2b,ymm4b,ymm0b);
    ymm1b = _FNMSUB(ymm2b,ymm5b,ymm1b);

    ymm0a = _FNMSUB(ymm3a,ymm5a,ymm0a);
    ymm1a = _FMADD(ymm3a,ymm4a,ymm1a);

    ymm0b = _FNMSUB(ymm3b,ymm5b,ymm0b);
    ymm1b = _FMADD(ymm3b,ymm4b,ymm1b);

    // store

    _mm256_store_ps(rr, ymm0a);
    _mm256_store_ps(ri, ymm1a);

    _mm256_store_ps(rr+8, ymm0b);
    _mm256_store_ps(ri+8, ymm1b);
}


inline void complex_split_mst_24( float const * ar, float const * ai,
                                  float const * br, float const * bi,
                                  float * rr, float * ri ) noexcept
{
    __m256 ymm0a;
    __m256 ymm1a;
    __m256 ymm2a;
    __m256 ymm5a;
    __m256 ymm4a;

    __m256 ymm0b;
    __m256 ymm1b;
    __m256 ymm2b;
    __m256 ymm5b;
    __m256 ymm4b;

    __m256 ymm0c;
    __m256 ymm1c;
    __m256 ymm2c;
    __m256 ymm5c;
    __m256 ymm4c;


    // rr += ar*br - ai*bi
    // ri += ar*bi + ai*br

    ymm2a = _mm256_load_ps(ar);
    ymm2b = _mm256_load_ps(ar+8);
    ymm2c = _mm256_load_ps(ar+16);

    ymm4a = _mm256_load_ps(br);
    ymm4b = _mm256_load_ps(br+8);
    ymm4c = _mm256_load_ps(br+16);

    ymm5a = _mm256_load_ps(bi);
    ymm5b = _mm256_load_ps(bi+8);
    ymm5c = _mm256_load_ps(bi+16);

    ymm0a = _mm256_mul_ps(ymm2a,ymm4a);
    ymm0b = _mm256_mul_ps(ymm2b,ymm4b);
    ymm0c = _mm256_mul_ps(ymm2c,ymm4c);

    ymm1a = _mm256_mul_ps(ymm2a,ymm5a);
    ymm1b = _mm256_mul_ps(ymm2b,ymm5b);
    ymm1c = _mm256_mul_ps(ymm2c,ymm5c);

    ymm2a = _mm256_load_ps(ai);
    ymm2b = _mm256_load_ps(ai+8);
    ymm2c = _mm256_load_ps(ai+16);

    ymm1a = _FMADD(ymm2a,ymm4a,ymm1a);
    ymm1b = _FMADD(ymm2b,ymm4b,ymm1b);
    ymm1c = _FMADD(ymm2c,ymm4c,ymm1c);

    ymm0a = _FNMSUB(ymm2a,ymm5a,ymm0a);
    ymm0b = _FNMSUB(ymm2b,ymm5b,ymm0b);
    ymm0c = _FNMSUB(ymm2c,ymm5c,ymm0c);

    // store

    _mm256_store_ps(rr, ymm0a);
    _mm256_store_ps(rr+8, ymm0b);
    _mm256_store_ps(rr+16, ymm0c);

    _mm256_store_ps(ri, ymm1a);
    _mm256_store_ps(ri+8, ymm1b);
    _mm256_store_ps(ri+16, ymm1c);
}



inline void complex_split_mad_24( float const * ar, float const * ai,
                                  float const * br, float const * bi,
                                  float * rr, float * ri ) noexcept
{
    __m256 ymm0a;
    __m256 ymm1a;
    __m256 ymm2a;
    __m256 ymm5a;
    __m256 ymm4a;

    __m256 ymm0b;
    __m256 ymm1b;
    __m256 ymm2b;
    __m256 ymm5b;
    __m256 ymm4b;

    __m256 ymm0c;
    __m256 ymm1c;
    __m256 ymm2c;
    __m256 ymm5c;
    __m256 ymm4c;


    // rr += ar*br - ai*bi
    // ri += ar*bi + ai*br

    ymm0a = _mm256_load_ps(rr);
    ymm0b = _mm256_load_ps(rr+8);
    ymm0c = _mm256_load_ps(rr+16);

    ymm1a = _mm256_load_ps(ri);
    ymm1b = _mm256_load_ps(ri+8);
    ymm1c = _mm256_load_ps(ri+16);

    ymm2a = _mm256_load_ps(ar);
    ymm2b = _mm256_load_ps(ar+8);
    ymm2c = _mm256_load_ps(ar+16);

    ymm4a = _mm256_load_ps(br);
    ymm4b = _mm256_load_ps(br+8);
    ymm4c = _mm256_load_ps(br+16);

    ymm5a = _mm256_load_ps(bi);
    ymm5b = _mm256_load_ps(bi+8);
    ymm5c = _mm256_load_ps(bi+16);

    ymm0a = _FMADD(ymm2a,ymm4a,ymm0a);
    ymm0b = _FMADD(ymm2b,ymm4b,ymm0b);
    ymm0c = _FMADD(ymm2c,ymm4c,ymm0c);

    ymm1a = _FNMSUB(ymm2a,ymm5a,ymm1a);
    ymm1b = _FNMSUB(ymm2b,ymm5b,ymm1b);
    ymm1c = _FNMSUB(ymm2c,ymm5c,ymm1c);

    ymm2a = _mm256_load_ps(ai);
    ymm2b = _mm256_load_ps(ai+8);
    ymm2c = _mm256_load_ps(ai+16);

    ymm1a = _FMADD(ymm2a,ymm4a,ymm1a);
    ymm1b = _FMADD(ymm2b,ymm4b,ymm1b);
    ymm1c = _FMADD(ymm2c,ymm4c,ymm1c);

    ymm0a = _FNMSUB(ymm2a,ymm5a,ymm0a);
    ymm0b = _FNMSUB(ymm2b,ymm5b,ymm0b);
    ymm0c = _FNMSUB(ymm2c,ymm5c,ymm0c);

    // store

    _mm256_store_ps(rr, ymm0a);
    _mm256_store_ps(rr+8, ymm0b);
    _mm256_store_ps(rr+16, ymm0c);

    _mm256_store_ps(ri, ymm1a);
    _mm256_store_ps(ri+8, ymm1b);
    _mm256_store_ps(ri+16, ymm1c);
}



inline void complex_split_mst( float const * ar, float const * ai,
                               float const * br, float const * bi,
                               float * rr, float * ri, std::size_t n ) noexcept
{
    typedef decltype(&complex_split_mst_8) fn_type;
    static fn_type dispatch[3] = { complex_split_mst_0,
                                   complex_split_mst_8,
                                   complex_split_mst_16 };

    n = (n+7) / 8;

    for ( std::size_t i = 0; i < n / 3; ++i )
    {
        complex_split_mst_24(ar,ai,br,bi,rr,ri);
        ar += 24; br += 24; rr += 24;
        ai += 24; bi += 24; ri += 24;
    }

    (dispatch[n%3])(ar,ai,br,bi,rr,ri);
}

inline void complex_split_mad( float const * ar, float const * ai,
                               float const * br, float const * bi,
                               float * rr, float * ri, std::size_t n ) noexcept
{
    typedef decltype(&complex_split_mad_8) fn_type;
    static fn_type dispatch[3] = { complex_split_mad_0,
                                   complex_split_mad_8,
                                   complex_split_mad_16 };

    n = (n+7) / 8;

    for ( std::size_t i = 0; i < n / 3; ++i )
    {
        complex_split_mad_24(ar,ai,br,bi,rr,ri);
        ar += 24; br += 24; rr += 24;
        ai += 24; bi += 24; ri += 24;
    }

    (dispatch[n%3])(ar,ai,br,bi,rr,ri);
}

}}} // namespace znn::fwd::host


#undef _FNMSUB
#undef _FMADD
