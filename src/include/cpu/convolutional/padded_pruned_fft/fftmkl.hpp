#pragma once

#include <mkl_dfti.h>

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

namespace znn { namespace fwd { namespace cpu {

#if defined(ZNN_USE_LONG_DOUBLE_PRECISION)

#elif defined(ZNN_USE_DOUBLE_PRECISION)

#define ZNN_DFTI_TYPE DFTI_DOUBLE

#else

#define ZNN_DFTI_TYPE DFTI_SINGLE

#endif

typedef DFTI_DESCRIPTOR_HANDLE fft_plan;

class padded_pruned_fft_transformer: public padded_pruned_fft_transformer_base
{
private:
    fft_plan ifwd1;
    fft_plan kfwd1;
    fft_plan bwd1;
    fft_plan t2, t3;
    fft_plan pt3;

public:
    ~padded_pruned_fft_transformer()
    {
        DftiFreeDescriptor(&ifwd1);
        DftiFreeDescriptor(&kfwd1);
        DftiFreeDescriptor(&bwd1);
        DftiFreeDescriptor(&t2);
        DftiFreeDescriptor(&t3);
        DftiFreeDescriptor(&pt3);
    }

    padded_pruned_fft_transformer( vec3i const & _im,
                                   vec3i const & _fs )
        : padded_pruned_fft_transformer_base(_im, _fs)
    {
        MKL_LONG status;

        // Out-of-place
        // Real to complex / complex to real along x direction
        // Repeated along z direction
        // Will need filter.y calls for each y
        {
            MKL_LONG strides_img[2] = { 0, isize[1] * isize[2] };
            MKL_LONG strides_ker[2] = { 0, ksize[1] * ksize[2] };
            MKL_LONG strides_res[2] = { 0, rsize[1] * rsize[2] };
            MKL_LONG strides_out[2] = { 0, csize[1] * csize[2] };

            status = DftiCreateDescriptor( &ifwd1, ZNN_DFTI_TYPE,
                                           DFTI_REAL, 1, asize[0] );
            status = DftiCreateDescriptor( &kfwd1, ZNN_DFTI_TYPE,
                                           DFTI_REAL, 1, asize[0] );
            status = DftiCreateDescriptor( &bwd1, ZNN_DFTI_TYPE,
                                           DFTI_REAL, 1, asize[0] );

            status = DftiSetValue( ifwd1 , DFTI_CONJUGATE_EVEN_STORAGE,
                                   DFTI_COMPLEX_COMPLEX );
            status = DftiSetValue( kfwd1 , DFTI_CONJUGATE_EVEN_STORAGE,
                                   DFTI_COMPLEX_COMPLEX );
            status = DftiSetValue( bwd1 , DFTI_CONJUGATE_EVEN_STORAGE,
                                   DFTI_COMPLEX_COMPLEX );

            status = DftiSetValue( ifwd1, DFTI_PLACEMENT, DFTI_NOT_INPLACE );
            status = DftiSetValue( kfwd1, DFTI_PLACEMENT, DFTI_NOT_INPLACE );
            status = DftiSetValue( bwd1 , DFTI_PLACEMENT, DFTI_NOT_INPLACE );

            status = DftiSetValue( ifwd1, DFTI_INPUT_STRIDES, strides_img );
            status = DftiSetValue( kfwd1, DFTI_INPUT_STRIDES, strides_ker );
            status = DftiSetValue( bwd1 , DFTI_INPUT_STRIDES, strides_out );

            status = DftiSetValue( ifwd1, DFTI_OUTPUT_STRIDES, strides_out );
            status = DftiSetValue( kfwd1, DFTI_OUTPUT_STRIDES, strides_out );
            status = DftiSetValue( bwd1 , DFTI_OUTPUT_STRIDES, strides_res );

            status = DftiSetValue( ifwd1, DFTI_NUMBER_OF_TRANSFORMS, isize[2] );
            status = DftiSetValue( kfwd1, DFTI_NUMBER_OF_TRANSFORMS, ksize[2] );
            status = DftiSetValue( bwd1 , DFTI_NUMBER_OF_TRANSFORMS, rsize[2] );

            status = DftiSetValue( ifwd1, DFTI_INPUT_DISTANCE, 1 );
            status = DftiSetValue( kfwd1, DFTI_INPUT_DISTANCE, 1 );
            status = DftiSetValue( bwd1 , DFTI_INPUT_DISTANCE, 1 );

            status = DftiSetValue( ifwd1, DFTI_OUTPUT_DISTANCE, 1 );
            status = DftiSetValue( kfwd1, DFTI_OUTPUT_DISTANCE, 1 );
            status = DftiSetValue( bwd1 , DFTI_OUTPUT_DISTANCE, 1 );

            status = DftiCommitDescriptor(ifwd1);
            status = DftiCommitDescriptor(kfwd1);
            status = DftiCommitDescriptor(bwd1);
        }

        // In-place
        // Complex to complex along y direction
        // Repeated along x direction
        // Will need filter.z calls for each z
        {
            MKL_LONG strides[2]  = { 0, csize[2] };

            status = DftiCreateDescriptor( &t2, ZNN_DFTI_TYPE,
                                           DFTI_COMPLEX, 1, csize[1] );

            status = DftiSetValue( t2, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX );
            status = DftiSetValue( t2, DFTI_PLACEMENT, DFTI_INPLACE );
            status = DftiSetValue( t2, DFTI_NUMBER_OF_TRANSFORMS, csize[0] );
            status = DftiSetValue( t2, DFTI_INPUT_STRIDES, strides );
            status = DftiSetValue( t2, DFTI_OUTPUT_STRIDES, strides );
            status = DftiSetValue( t2, DFTI_INPUT_DISTANCE, csize[2]*csize[1] );
            status = DftiSetValue( t2, DFTI_OUTPUT_DISTANCE, csize[2]*csize[1] );

            status = DftiCommitDescriptor(t2);
        }


        // In-place
        // Complex to complex along z direction
        // Repeated along x and y directions
        // Single call
        {
            MKL_LONG strides[2]  = { 0, 1 };

            status = DftiCreateDescriptor( &t3, ZNN_DFTI_TYPE,
                                           DFTI_COMPLEX, 1, csize[2] );

            status = DftiSetValue( t3, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX );
            status = DftiSetValue( t3, DFTI_PLACEMENT, DFTI_INPLACE );
            status = DftiSetValue( t3, DFTI_NUMBER_OF_TRANSFORMS, csize[0]*csize[1] );
            status = DftiSetValue( t3, DFTI_INPUT_STRIDES, strides );
            status = DftiSetValue( t3, DFTI_OUTPUT_STRIDES, strides );
            status = DftiSetValue( t3, DFTI_INPUT_DISTANCE, csize[2] );
            status = DftiSetValue( t3, DFTI_OUTPUT_DISTANCE, csize[2] );

            status = DftiCommitDescriptor(t3);

            status = DftiCreateDescriptor( &pt3, ZNN_DFTI_TYPE,
                                           DFTI_COMPLEX, 1, csize[2] );

            status = DftiSetValue( pt3, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX );
            status = DftiSetValue( pt3, DFTI_PLACEMENT, DFTI_INPLACE );
            status = DftiSetValue( pt3, DFTI_NUMBER_OF_TRANSFORMS, csize[1] );
            status = DftiSetValue( pt3, DFTI_INPUT_STRIDES, strides );
            status = DftiSetValue( pt3, DFTI_OUTPUT_STRIDES, strides );
            status = DftiSetValue( pt3, DFTI_INPUT_DISTANCE, csize[2] );
            status = DftiSetValue( pt3, DFTI_OUTPUT_DISTANCE, csize[2] );

            status = DftiCommitDescriptor(pt3);

        }
    }

    void forward_kernel( real* rp, void* cpv )
    {
        MKL_LONG status;

        real* cp = reinterpret_cast<real*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(real)*2);

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < ksize[1]; ++i )
        {
            status = DftiComputeForward(kfwd1,
                                        rp + ksize[2] * i,
                                        cp + csize[2] * i * 2 );
        }

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < ksize[2]; ++i )
        {
            status = DftiComputeForward( t2, cp + i * 2, cp + i * 2 );
        }

        // In-place complex to complex along z-direction
        status = DftiComputeForward( t3, cp, cp );
    }

    void forward_image( real* rp, void* cpv )
    {
        MKL_LONG status;

        real* cp = reinterpret_cast<real*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(real)*2);

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < isize[1]; ++i )
        {
            status = DftiComputeForward(ifwd1,
                                        rp + isize[2] * i,
                                        cp + csize[2] * i * 2 );
        }

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < isize[2]; ++i )
        {
            status = DftiComputeForward( t2, cp + i * 2, cp + i * 2 );
        }

        // In-place complex to complex along z-direction
        status = DftiComputeForward( t3, cp, cp );
    }

    void backward( void* cpv, real* rp )
    {
        MKL_LONG status;

        real* cp = reinterpret_cast<real*>(cpv);
        // In-place complex to complex along z-direction
        status = DftiComputeBackward( t3, cp, cp );

        // In-place complex to complex along y-direction
        // Care only about last rsize[2]
        long_t zOff = ksize[2] - 1;
        for ( long_t i = 0; i < rsize[2]; ++i )
        {
            status = DftiComputeBackward( t2,
                                          cp + (i + zOff) * 2,
                                          cp + (i + zOff) * 2 );
        }

        // Out-of-place complex to real along x-direction
        // Care only about last rsize[1] and rsize[2]
        long_t yOff = ksize[1] - 1;
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            status = DftiComputeBackward( bwd1,
                                          cp + (csize[2] * ( i + yOff ) + zOff) * 2,
                                          rp + rsize[2] * i );
        }
    }


    void parallel_forward_kernel( task_package & handle, real* rp, void* cpv )
    {
        real* cp = reinterpret_cast<real*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(real)*2);

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < ksize[1]; ++i )
        {
            handle.add_task( [rp, cp, i, this](void*) {
                    MKL_LONG status
                        = DftiComputeForward(this->kfwd1,
                                             rp + this->ksize[2] * i,
                                             cp + this->csize[2] * i * 2 );
                });
        }

        handle.execute();


        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < ksize[2]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    MKL_LONG status
                        = DftiComputeForward( this->t2,
                                              cp + i * 2,
                                              cp + i * 2 );
                });
        }

        handle.execute();

        // In-place complex to complex along z-direction
        for ( long_t i = 0; i < csize[0]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    MKL_INT status
                        = DftiComputeForward( this->t3,
                                              cp + i * this->csize[2]*this->csize[1],
                                              cp + i * this->csize[2]*this->csize[1] );
                });
        }

        handle.execute();
    }

    void parallel_forward_image( task_package & handle, real* rp, void* cpv )
    {
        real* cp = reinterpret_cast<real*>(cpv);
        std::memset(cp, 0, csize[0]*csize[1]*csize[2]*sizeof(real)*2);

        // Out-of-place real to complex along x-direction
        for ( long_t i = 0; i < isize[1]; ++i )
        {
            handle.add_task( [rp, cp, i, this](void*) {
                    MKL_LONG status
                        = DftiComputeForward(this->ifwd1,
                                             rp + this->isize[2] * i,
                                             cp + this->csize[2] * i * 2 );
                });
        }

        handle.execute();

        // In-place complex to complex along y-direction
        for ( long_t i = 0; i < isize[2]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    MKL_LONG status
                        = DftiComputeForward( this->t2,
                                              cp + i * 2,
                                              cp + i * 2 );
                });
        }

        handle.execute();

        // In-place complex to complex along z-direction
        for ( long_t i = 0; i < csize[0]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    MKL_INT status
                        = DftiComputeForward( this->t3,
                                              cp + i * this->csize[2]*this->csize[1],
                                              cp + i * this->csize[2]*this->csize[1] );
                });
        }

        handle.execute();
    }

    void parallel_backward( task_package & handle, void* cpv, real* rp )
    {
        real* cp = reinterpret_cast<real*>(cpv);

        // In-place complex to complex along z-direction
        for ( long_t i = 0; i < csize[0]; ++i )
        {
            handle.add_task( [cp, i, this](void*) {
                    MKL_LONG status =
                        DftiComputeBackward( this->t3,
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
            handle.add_task( [zOff, cp, i, this](void*) {
                    MKL_LONG status
                        = DftiComputeBackward( this->t2,
                                               cp + (i + zOff) * 2,
                                               cp + (i + zOff) * 2 );
                });
        }

        handle.execute();

        // Out-of-place complex to real along x-direction
        // Care only about last rsize[1] and rsize[2]
        long_t yOff = ksize[1] - 1;
        for ( long_t i = 0; i < rsize[1]; ++i )
        {
            handle.add_task( [zOff, yOff, rp, cp, i, this](void*) {
                    MKL_LONG status
                        = DftiComputeBackward( this->bwd1,
                                               cp + (this->csize[2] * ( i + yOff ) + zOff) * 2,
                                               rp + this->rsize[2] * i );
                });
        }

        handle.execute();

    }

};

}}} // namespace znn::fwd::cpu
