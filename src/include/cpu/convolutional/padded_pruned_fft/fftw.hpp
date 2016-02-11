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

#include "../../../assert.hpp"
#include "../../../types.hpp"
#include "../../../memory.hpp"
#include "../../utils/task_package.hpp"
#include "base.hpp"

#ifndef ZNN_FFTW_PLANNING_MODE
#  define ZNN_FFTW_PLANNING_MODE (FFTW_ESTIMATE)
#endif

namespace znn { namespace fwd { namespace cpu {

#if defined(ZNN_USE_LONG_DOUBLE_PRECISION)

#  define FFT_DESTROY_PLAN fftwl_destroy_plan
#  define FFT_CLEANUP      fftwl_cleanup
#  define FFT_PLAN_C2R     fftwl_plan_dft_c2r_3d
#  define FFT_PLAN_R2C     fftwl_plan_dft_r2c_3d

#  define FFT_PLAN_MANY_DFT fftwl_plan_many_dft
#  define FFT_PLAN_MANY_R2C fftwl_plan_many_dft_r2c
#  define FFT_PLAN_MANY_C2R fftwl_plan_many_dft_c2r

#  define FFT_EXECUTE_DFT_R2C fftwl_execute_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftwl_execute_dft_c2r
#  define FFT_EXECUTE_DFT     fftwl_execute_dft

typedef fftwl_plan    fft_plan   ;
typedef fftwl_complex fft_complex;

#elif defined(ZNN_USE_DOUBLE_PRECISION)

#  define FFT_DESTROY_PLAN fftw_destroy_plan
#  define FFT_CLEANUP      fftw_cleanup
#  define FFT_PLAN_C2R     fftw_plan_dft_c2r_3d
#  define FFT_PLAN_R2C     fftw_plan_dft_r2c_3d

#  define FFT_PLAN_MANY_DFT fftw_plan_many_dft
#  define FFT_PLAN_MANY_R2C fftw_plan_many_dft_r2c
#  define FFT_PLAN_MANY_C2R fftw_plan_many_dft_c2r

#  define FFT_EXECUTE_DFT_R2C fftw_execute_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftw_execute_dft_c2r
#  define FFT_EXECUTE_DFT     fftw_execute_dft

typedef fftw_plan    fft_plan   ;
typedef fftw_complex fft_complex;

#else

#  define FFT_DESTROY_PLAN fftwf_destroy_plan
#  define FFT_CLEANUP      fftwf_cleanup
#  define FFT_PLAN_C2R     fftwf_plan_dft_c2r_3d
#  define FFT_PLAN_R2C     fftwf_plan_dft_r2c_3d

#  define FFT_PLAN_MANY_DFT fftwf_plan_many_dft
#  define FFT_PLAN_MANY_R2C fftwf_plan_many_dft_r2c
#  define FFT_PLAN_MANY_C2R fftwf_plan_many_dft_c2r

#  define FFT_EXECUTE_DFT_R2C fftwf_execute_dft_r2c
#  define FFT_EXECUTE_DFT_C2R fftwf_execute_dft_c2r
#  define FFT_EXECUTE_DFT     fftwf_execute_dft

typedef fftwf_plan    fft_plan   ;
typedef fftwf_complex fft_complex;

#endif

class padded_pruned_fft_transformer: public padded_pruned_fft_transformer_base
{
private:
    fft_plan ifwd1;
    fft_plan kfwd1;
    fft_plan fwd2, fwd3;

    fft_plan bwd1, bwd2, bwd3;

    // for parallel execution
    fft_plan pbwd1, pfwd3;

public:
    ~padded_pruned_fft_transformer()
    {
        FFT_DESTROY_PLAN(ifwd1);
        FFT_DESTROY_PLAN(kfwd1);
        FFT_DESTROY_PLAN(fwd2);
        FFT_DESTROY_PLAN(fwd3);
        FFT_DESTROY_PLAN(bwd1);
        FFT_DESTROY_PLAN(bwd2);
        FFT_DESTROY_PLAN(bwd3);
        FFT_DESTROY_PLAN(pbwd1);
        FFT_DESTROY_PLAN(pfwd3);
    }

    padded_pruned_fft_transformer( vec3i const & _im,
                                   vec3i const & _fs )
        : padded_pruned_fft_transformer_base(_im, _fs)
    {
        host_array<real>  rp
            = get_array<real>(asize[0]*asize[1]*asize[2]);

        host_array<fft_complex> cp
            = get_array<fft_complex>(csize[0]*csize[1]*csize[2]);

        // Out-of-place
        // Real to complex / complex to real along x direction
        // Repeated along z direction
        // Will need filter.y calls for each y
        {
            int len[]     = { static_cast<int>(asize[0]) };

            ifwd1 = FFT_PLAN_MANY_R2C(
                1, len,
                isize[2], // How many
                rp.get(), NULL,
                isize[2] * isize[1], // Input stride
                1, // Input distance
                cp.get(), NULL,
                csize[2] * csize[1], // Output stride
                1, // Output distance
                ZNN_FFTW_PLANNING_MODE );

            kfwd1 = FFT_PLAN_MANY_R2C(
                1, len,
                ksize[2], // How many
                rp.get(), NULL,
                ksize[2] * ksize[1], // Input stride
                1, // Input distance
                cp.get(), NULL,
                csize[2] * csize[1], // Output stride
                1, // Output distance
                ZNN_FFTW_PLANNING_MODE );

            bwd3 = FFT_PLAN_MANY_C2R(
                1, len,
                rsize[2], // How many
                cp.get(), NULL,
                csize[2] * csize[1], // Input stride
                1, // Input distance
                rp.get(), NULL,
                rsize[2] * rsize[1], // Output stride
                1, // Output distance
                ZNN_FFTW_PLANNING_MODE );
        }

        // In-place
        // Complex to complex along y direction
        // Repeated along x direction
        // Will need filter.z/image.z calls for each z
        {
            int n[]     = { static_cast<int>(csize[1]) };
            int howmany = static_cast<int>(csize[0]);
            int stride  = static_cast<int>(csize[2]);
            int dist    = static_cast<int>(csize[2]*csize[1]);

            fwd2 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                      cp.get(), NULL, stride, dist,
                                      cp.get(), NULL, stride, dist,
                                      FFTW_FORWARD, ZNN_FFTW_PLANNING_MODE );

            bwd2 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                      cp.get(), NULL, stride, dist,
                                      cp.get(), NULL, stride, dist,
                                      FFTW_BACKWARD, ZNN_FFTW_PLANNING_MODE );
        }


        // In-place
        // Complex to complex along z direction
        // Repeated along x and y directions
        // Single call
        {
            int n[]     = { static_cast<int>(csize[2]) };
            int howmany = static_cast<int>(csize[0]*csize[1]);
            int stride  = static_cast<int>(1);
            int dist    = static_cast<int>(csize[2]);

            fwd3 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                      cp.get(), NULL, stride, dist,
                                      cp.get(), NULL, stride, dist,
                                      FFTW_FORWARD, ZNN_FFTW_PLANNING_MODE );

            bwd1 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                      cp.get(), NULL, stride, dist,
                                      cp.get(), NULL, stride, dist,
                                      FFTW_BACKWARD, ZNN_FFTW_PLANNING_MODE );

            // each thread does it along y direction
            // separate threads for separate x coordinate

            howmany = static_cast<int>(csize[1]);

            pfwd3 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                       cp.get(), NULL, stride, dist,
                                       cp.get(), NULL, stride, dist,
                                       FFTW_FORWARD, ZNN_FFTW_PLANNING_MODE );

            pbwd1 = FFT_PLAN_MANY_DFT( 1, n, howmany,
                                       cp.get(), NULL, stride, dist,
                                       cp.get(), NULL, stride, dist,
                                       FFTW_BACKWARD, ZNN_FFTW_PLANNING_MODE );
        }

    }

    void forward_kernel( real* rp, void* cpv )
    {
        fft_complex* cp = reinterpret_cast<fft_complex*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(fft_complex));

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < ksize[1]; ++i )
        {
            FFT_EXECUTE_DFT_R2C( kfwd1, rp + ksize[2] * i, cp + csize[2] * i );
        }

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < ksize[2]; ++i )
        {
            FFT_EXECUTE_DFT( fwd2, cp + i, cp + i );
        }

        // In-place complex to complex along z-direction
        FFT_EXECUTE_DFT( fwd3, cp, cp );
        // for ( long_t i = 0; i < csize[0]; ++i )
        // {
        //     FFT_EXECUTE_DFT( pfwd3,
        //                      cp + i * csize[2]*csize[1],
        //                      cp + i * csize[2]*csize[1] );

        // }
    }

    void forward_image( real* rp, void* cpv )
    {
        fft_complex* cp = reinterpret_cast<fft_complex*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(fft_complex));

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < isize[1]; ++i )
        {
            FFT_EXECUTE_DFT_R2C( ifwd1, rp + isize[2] * i, cp + csize[2] * i );
        }

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < isize[2]; ++i )
        {
            FFT_EXECUTE_DFT( fwd2, cp + i, cp + i );
        }

        // In-place complex to complex along z-direction
        FFT_EXECUTE_DFT( fwd3, cp, cp );
        // for ( long_t i = 0; i < csize[0]; ++i )
        // {
        //     FFT_EXECUTE_DFT( pfwd3,
        //                      cp + i * csize[2]*csize[1],
        //                      cp + i * csize[2]*csize[1] );

        // }

    }

    void backward( void* cpv, real* rp )
    {
        fft_complex* cp = reinterpret_cast<fft_complex*>(cpv);
        // In-place complex to complex along z-direction
        FFT_EXECUTE_DFT( bwd1, cp, cp );
        // for ( long_t i = 0; i < csize[0]; ++i )
        // {
        //     FFT_EXECUTE_DFT( pbwd1,
        //                      cp + i * csize[2]*csize[1],
        //                      cp + i * csize[2]*csize[1] );

        // }


        // In-place complex to complex along y-direction
        // Care only about last rsize[2]
        long_t zOff = ksize[2] - 1;
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            FFT_EXECUTE_DFT( bwd2, cp + i + zOff, cp + i + zOff );
        }

        // Out-of-place complex to real along x-direction
        // Care only about last rsize[1] and rsize[2]
        long_t yOff = ksize[1] - 1;
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            FFT_EXECUTE_DFT_C2R( bwd3,
                                 cp + csize[2] * ( i + yOff ) + zOff,
                                 rp + rsize[2] * i );
        }
    }

    void parallel_forward_kernel( task_package & handle, real* rp, void* cpv )
    {
        fft_complex* cp = reinterpret_cast<fft_complex*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(fft_complex));

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < ksize[1]; ++i )
        {
            handle.add_task( [rp, cp, i, this](void*) {
                    FFT_EXECUTE_DFT_R2C( this->kfwd1,
                                         rp + this->ksize[2] * i,
                                         cp + this->csize[2] * i );
                });
        }

        handle.execute();

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < ksize[2]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    FFT_EXECUTE_DFT( this->fwd2, cp + i, cp + i );
                });
        }

        handle.execute();

        // In-place complex to complex along z-direction
        for ( long_t i = 0; i < csize[0]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    FFT_EXECUTE_DFT( this->pfwd3,
                                     cp + i * this->csize[2]*this->csize[1],
                                     cp + i * this->csize[2]*this->csize[1] );
                });

        }

        handle.execute();

    }

    void parallel_forward_image( task_package & handle, real* rp, void* cpv )
    {
        fft_complex* cp = reinterpret_cast<fft_complex*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(fft_complex));

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < isize[1]; ++i )
        {
            handle.add_task( [rp, cp, i, this](void*) {
                    FFT_EXECUTE_DFT_R2C( this->ifwd1,
                                         rp + this->isize[2] * i,
                                         cp + this->csize[2] * i );
                });
        }

        handle.execute();

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < isize[2]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    FFT_EXECUTE_DFT( this->fwd2, cp + i, cp + i );
                });
        }

        handle.execute();

        // In-place complex to complex along z-direction
        for ( long_t i = 0; i < csize[0]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    FFT_EXECUTE_DFT( this->pfwd3,
                                     cp + i * this->csize[2]*this->csize[1],
                                     cp + i * this->csize[2]*this->csize[1] );
                });

        }

        handle.execute();
    }

    void parallel_backward( task_package & handle, void* cpv, real* rp )
    {
        fft_complex* cp = reinterpret_cast<fft_complex*>(cpv);
        // In-place complex to complex along z-direction
        for ( long_t i = 0; i < csize[0]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    FFT_EXECUTE_DFT( this->pbwd1,
                                     cp + i * this->csize[2]*this->csize[1],
                                     cp + i * this->csize[2]*this->csize[1] );
                });
        }

        handle.execute();

        // In-place complex to complex along y-direction
        // Care only about last rsize[2]
        long_t zOff = ksize[2] - 1;
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            handle.add_task( [&zOff, cp, i, this](void*) {
                    FFT_EXECUTE_DFT( this->bwd2, cp + i + zOff, cp + i + zOff );
                });
        }

        handle.execute();

        // Out-of-place complex to real along x-direction
        // Care only about last rsize[1] and rsize[2]
        long_t yOff = ksize[1] - 1;
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            handle.add_task( [&zOff, &yOff, rp, cp, i, this](void*) {
                    FFT_EXECUTE_DFT_C2R( this->bwd3,
                                         cp + this->csize[2] * ( i + yOff ) + zOff,
                                         rp + this->rsize[2] * i );
                });
        }

        handle.execute();
    }


};

}}} // namespace znn::fwd::cpu


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
