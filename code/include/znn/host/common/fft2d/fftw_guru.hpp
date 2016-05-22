#pragma once

#ifdef ZNN_USE_MKL_FFTW
#  include <fftw/fftw3.h>
#else
#  include <fftw3.h>
#endif

#include <map>
#include <iostream>
#include <unordered_map>
#include <type_traits>
#include <mutex>
#include <zi/utility/singleton.hpp>
#include <tbb/tbb.h>

#include "znn/assert.hpp"
#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/common/fft2d/base.hpp"

#ifndef ZNN_FFTW_PLANNING_MODE
#  define ZNN_FFTW_PLANNING_MODE (FFTW_ESTIMATE)
#endif

namespace znn { namespace fwd { namespace host {

#if defined(ZNN_USE_LONG_DOUBLE_PRECISION)

#  define FFT_DESTROY_PLAN fftwl_destroy_plan
#  define FFT_CLEANUP      fftwl_cleanup
#  define FFT_PLAN_C2R     fftwl_plan_guru64_split_dft_c2r
#  define FFT_PLAN_R2C     fftwl_plan_guru64_split_dft_r2c
#  define FFT_PLAN_DFT     fftwl_plan_guru64_split_dft

#  define FFT_EXECUTE_DFT_R2C fftwl_execute_split_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftwl_execute_split_dft_c2r
#  define FFT_EXECUTE_DFT     fftwl_execute_split_dft

typedef fftwl_plan    fft_plan   ;
typedef fftwl_complex fft_complex;

#elif defined(ZNN_USE_DOUBLE_PRECISION)

#  define FFT_DESTROY_PLAN fftw_destroy_plan
#  define FFT_CLEANUP      fftw_cleanup
#  define FFT_PLAN_C2R     fftw_plan_guru64_split_dft_c2r
#  define FFT_PLAN_R2C     fftw_plan_guru64_split_dft_r2c
#  define FFT_PLAN_DFT     fftw_plan_guru64_split_dft

#  define FFT_EXECUTE_DFT_R2C fftw_execute_split_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftw_execute_split_dft_c2r
#  define FFT_EXECUTE_DFT     fftw_execute_split_dft

typedef fftw_plan    fft_plan   ;
typedef fftw_complex fft_complex;

#else

#  define FFT_DESTROY_PLAN fftwf_destroy_plan
#  define FFT_CLEANUP      fftwf_cleanup
#  define FFT_PLAN_C2R     fftwf_plan_guru64_split_dft_c2r
#  define FFT_PLAN_R2C     fftwf_plan_guru64_split_dft_r2c
#  define FFT_PLAN_DFT     fftwf_plan_guru64_split_dft

#  define FFT_EXECUTE_DFT_R2C fftwf_execute_split_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftwf_execute_split_dft_c2r
#  define FFT_EXECUTE_DFT     fftwf_execute_split_dft

typedef fftwf_plan    fft_plan   ;
typedef fftwf_complex fft_complex;

#endif

class padded_pruned_fft2d_transformer: public padded_pruned_fft2d_transformer_base
{
private:
    fft_plan ifwd1;
    fft_plan kfwd1;
    fft_plan bwd2;

    fft_plan fwdbwd;

public:
    ~padded_pruned_fft2d_transformer()
    {
        FFT_DESTROY_PLAN(ifwd1);
        FFT_DESTROY_PLAN(kfwd1);
        FFT_DESTROY_PLAN(bwd2);
        FFT_DESTROY_PLAN(fwdbwd);
    }

    padded_pruned_fft2d_transformer( vec2i const & _im,
                                     vec2i const & _fs )
        : padded_pruned_fft2d_transformer_base(_im, _fs)
    {
        host_tensor<real,2> rp(asize);
        host_tensor<real,1> cp(padded_len*2);

        // Out-of-place
        // Real to complex / complex to real along x direction
        // Repeated along z direction
        // Will need filter.y calls for each y
        {
            fftw_iodim64 dims[1];
            fftw_iodim64 howmany_dims[1];

            dims[0].n  = asize[0];
            dims[0].is = isize[0]; // variable
            dims[0].os = csize[0];

            howmany_dims[0].n  = isize[1]; // variable
            howmany_dims[0].is = 1;
            howmany_dims[0].os = 1;

            ifwd1 = FFT_PLAN_R2C(
                1, dims,
                1, howmany_dims,
                rp.data(), cp.data(), cp.data() + padded_len,
                ZNN_FFTW_PLANNING_MODE);

            dims[0].is = ksize[1];
            howmany_dims[0].n = ksize[1];

            kfwd1 = FFT_PLAN_R2C(
                1, dims,
                1, howmany_dims,
                rp.data(), cp.data(), cp.data() + padded_len,
                ZNN_FFTW_PLANNING_MODE);

            dims[0].n  = asize[0];
            dims[0].is = csize[0];
            dims[0].os = rsize[0];

            howmany_dims[0].n  = rsize[1];
            howmany_dims[0].is = 1;
            howmany_dims[0].os = 1;

            bwd2 = FFT_PLAN_C2R(
                1, dims,
                1, howmany_dims,
                cp.data(), cp.data() + padded_len, rp.data(),
                ZNN_FFTW_PLANNING_MODE);
        }

        // In-place
        // Complex to complex along y direction
        // Single call
        {
            fftw_iodim64 dims[1];
            fftw_iodim64 howmany_dims[1];

            dims[0].n  = csize[1];
            dims[0].is = 1;
            dims[0].os = 1;

            howmany_dims[0].n  = csize[0]; // variable
            howmany_dims[0].is = csize[1];
            howmany_dims[0].os = csize[1];

            fwdbwd = FFT_PLAN_DFT(
                1, dims,
                1, howmany_dims,
                cp.data(), cp.data() + padded_len,
                cp.data(), cp.data() + padded_len,
                ZNN_FFTW_PLANNING_MODE);
        }

    }

    void forward_kernel( real* rp, void* cpv )
    {
        real* cp = reinterpret_cast<real*>(cpv);
        std::memset(cp, 0, 2*padded_len*sizeof(real));

        FFT_EXECUTE_DFT_R2C( kfwd1, rp, cp, cp + padded_len );
        FFT_EXECUTE_DFT( fwdbwd, cp, cp + padded_len, cp, cp + padded_len );
    }

    void forward_image( real* rp, void* cpv )
    {
        real* cp = reinterpret_cast<real*>(cpv);
        std::memset(cp, 0, 2*padded_len*sizeof(real));

        FFT_EXECUTE_DFT_R2C( ifwd1, rp, cp, cp + padded_len );
        FFT_EXECUTE_DFT( fwdbwd, cp, cp + padded_len, cp, cp + padded_len );
    }

    void backward( void* cpv, real* rp )
    {
        real* cp = reinterpret_cast<real*>(cpv);
        // In-place complex to complex along z-direction
        FFT_EXECUTE_DFT( fwdbwd, cp + padded_len, cp, cp + padded_len, cp );

        long_t yOff = ksize[1] - 1;
        FFT_EXECUTE_DFT_C2R( bwd2, cp + yOff, cp + yOff + padded_len, rp );
    }

};

}}} // namespace znn::fwd::host


#undef FFT_DESTROY_PLAN
#undef FFT_CLEANUP
#undef FFT_PLAN_R2C
#undef FFT_PLAN_C2R

#undef FFT_PLAN_MANY_DFT
#undef FFT_PLAN_MANY_R2C
#undef FFT_PLAN_MANY_C2R

#undef FFT_EXECUTE_DFT_R2C
#undef FFT_EXECUTE_DFT_C2R
#undef FFT_EXECUTE_DFT
