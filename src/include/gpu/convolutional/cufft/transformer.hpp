#pragma once

#include <cufft.h>
#include "../../../types.hpp"
#include "../../../assert.hpp"
#include "../../utils.hpp"
#include "../../memory.hpp"

namespace znn { namespace fwd { namespace gpu {


class cufft_forward_transformer
{
private:
    cufftHandle full_handle   ;
    cufftHandle partial_handle;

    long_t full_in_stride ;
    long_t full_out_stride;
    long_t num_full       ;
    long_t has_partial    ;

    long_t partial_in_stride ;
    long_t partial_out_stride;

public:
    ~cufft_forward_transformer()
    {
        checkCUFFT( cufftDestroy(full_handle) );
        if ( has_partial )
        {
            checkCUFFT( cufftDestroy(partial_handle) );
        }
    }

    cufft_forward_transformer( vec3i const & is, long_t n )
    {
        long_t max_elements = (2 << 22);

        vec3i cs = is;
        cs[2] /= 2; cs[2] += 1;

        STRONG_ASSERT(is[0]*is[1]*is[2] <= max_elements);
        long_t elements = is[0] * is[1] * is[2];

        long_t max_n = max_elements / elements;

        long_t n_full = std::min(n, max_n);

        num_full = n / n_full;

        full_in_stride  = is[0]*is[1]*is[2]*n_full;
        full_out_stride = cs[0]*cs[1]*cs[2]*n_full;

        int dims[3] = { static_cast<int>(is[0]),
                        static_cast<int>(is[1]),
                        static_cast<int>(is[2]) };

        int rdist   = static_cast<int>(is[0]*is[1]*is[2]);
        int cdist   = static_cast<int>(cs[0]*cs[1]*cs[2]);

        checkCUFFT( cufftPlanMany(&full_handle, 3, dims, NULL, 0,
                                  rdist, NULL, 0,
                                  cdist, CUFFT_R2C,
                                  static_cast<int>(n_full)) );
        has_partial = n % n_full;

        partial_in_stride  = is[0]*is[1]*is[2]*has_partial;
        partial_out_stride = cs[0]*cs[1]*cs[2]*has_partial;

        if ( has_partial )
        {
            checkCUFFT( cufftPlanMany(&partial_handle, 3, dims, NULL, 0,
                                      rdist, NULL, 0,
                                      cdist, CUFFT_R2C,
                                      static_cast<int>(has_partial)) );
        }
    }

    void forward( float* in, cuComplex* out ) const
    {
        checkCUFFT(cufftExecR2C(full_handle,in,out));

        if ( num_full > 1 )
        {
            float* oin = in;

            in  += full_in_stride;
            out += full_out_stride;

            auto out_mem = get_device_array<cuComplex>(full_out_stride);

            for ( long_t i = 1; i < num_full; ++i )
            {
                cudaMemcpy(oin, in, full_in_stride*sizeof(float),
                           cudaMemcpyDeviceToDevice);

                checkCUFFT(cufftExecR2C(full_handle,oin,out_mem.get()));

                cudaMemcpy(out, out_mem.get(), full_out_stride*sizeof(cuComplex),
                           cudaMemcpyDeviceToDevice);

                in  += full_in_stride;
                out += full_out_stride;
            }

            if ( has_partial )
            {
                cudaMemcpy(oin, in, partial_in_stride*sizeof(float),
                           cudaMemcpyDeviceToDevice);

                checkCUFFT(cufftExecR2C(partial_handle,oin,out_mem.get()));

                cudaMemcpy(out, out_mem.get(), partial_out_stride*sizeof(cuComplex),
                           cudaMemcpyDeviceToDevice);
            }
        }
    }

};


class cufft_backward_transformer
{
private:
    cufftHandle full_handle   ;
    cufftHandle partial_handle;

    long_t full_in_stride ;
    long_t full_out_stride;
    long_t num_full       ;
    long_t has_partial    ;

    long_t partial_in_stride ;
    long_t partial_out_stride;

public:
    ~cufft_backward_transformer()
    {
        checkCUFFT( cufftDestroy(full_handle) );
        if ( has_partial )
        {
            checkCUFFT( cufftDestroy(partial_handle) );
        }
    }

    cufft_backward_transformer( vec3i const & is, long_t n )
    {

        long_t max_elements = (2 << 22);

        vec3i cs = is;
        cs[2] /= 2; cs[2] += 1;

        STRONG_ASSERT(is[0]*is[1]*is[2] <= max_elements);
        long_t elements = is[0] * is[1] * is[2]; // for alignment

        long_t max_n = max_elements / elements;

        long_t n_full = std::min(n, max_n);

        num_full = n / n_full;

        full_in_stride  = cs[0]*cs[1]*cs[2]*n_full;
        full_out_stride = is[0]*is[1]*is[2]*n_full;

        int dims[3] = { static_cast<int>(is[0]),
                        static_cast<int>(is[1]),
                        static_cast<int>(is[2]) };

        int rdist   = static_cast<int>(is[0]*is[1]*is[2]);
        int cdist   = static_cast<int>(cs[0]*cs[1]*cs[2]);

        checkCUFFT( cufftPlanMany(&full_handle, 3, dims, NULL, 0,
                                  cdist, NULL, 0,
                                  rdist, CUFFT_C2R,
                                  static_cast<int>(n_full)) );
        has_partial = n % n_full;

        partial_in_stride  = cs[0]*cs[1]*cs[2]*has_partial;
        partial_out_stride = is[0]*is[1]*is[2]*has_partial;

        if ( has_partial )
        {
            checkCUFFT( cufftPlanMany(&partial_handle, 3, dims, NULL, 0,
                                      cdist, NULL, 0,
                                      rdist, CUFFT_C2R,
                                      static_cast<int>(has_partial)) );
        }
    }

    void backward( cuComplex* in, float* out ) const
    {
        checkCUFFT(cufftExecC2R(full_handle,in,out));

        if ( num_full > 1 )
        {
            cuComplex* oin = in;

            in  += full_in_stride;
            out += full_out_stride;

            auto out_mem = get_device_array<float>(full_out_stride);

            for ( long_t i = 1; i < num_full; ++i )
            {
                cudaMemcpy(oin, in, full_in_stride*sizeof(cuComplex),
                           cudaMemcpyDeviceToDevice);

                checkCUFFT(cufftExecC2R(full_handle,oin,out_mem.get()));

                cudaMemcpy(out, out_mem.get(), full_out_stride*sizeof(float),
                           cudaMemcpyDeviceToDevice);

                in  += full_in_stride;
                out += full_out_stride;
            }

            if ( has_partial )
            {
                cudaMemcpy(oin, in, partial_in_stride*sizeof(cuComplex),
                           cudaMemcpyDeviceToDevice);

                checkCUFFT(cufftExecC2R(partial_handle,oin,out_mem.get()));

                cudaMemcpy(out, out_mem.get(), partial_out_stride*sizeof(float),
                           cudaMemcpyDeviceToDevice);
            }
        }
    }

};

}}} // namespace znn::fwd::gpu
