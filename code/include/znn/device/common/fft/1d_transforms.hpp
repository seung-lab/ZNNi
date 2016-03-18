#pragma once

#include "znn/types.hpp"
#include "znn/device/common/utils.hpp"

#include <cufft.h>

namespace znn { namespace fwd { namespace device {

class cufft_1d_r2c_transformer
{
private:
    cufftHandle fwd_full, bwd_full;
    cufftHandle fwd_part, bwd_part;

    long_t n_full     ;
    long_t has_partial;

    long_t delta      ;
    size_t workspace_size_ = 0;

public:
    long_t workspace_size() const
    {
        return static_cast<long_t>(workspace_size_);
    }

    ~cufft_1d_r2c_transformer()
    {
        checkCUFFT( cufftDestroy(fwd_full) );
        checkCUFFT( cufftDestroy(fwd_part) );
        checkCUFFT( cufftDestroy(bwd_full) );
        checkCUFFT( cufftDestroy(bwd_part) );
    }

    cufft_1d_r2c_transformer( long_t l, long_t n )
    {
        int   dims[1] = { static_cast<int>(l) };
        size_t ws;

        long_t clen = l/2 + 1;

        checkCUFFT( cufftCreate(&fwd_full) );
        checkCUFFT( cufftCreate(&fwd_part) );
        checkCUFFT( cufftCreate(&bwd_full) );
        checkCUFFT( cufftCreate(&bwd_part) );

        checkCUFFT( cufftSetAutoAllocation(fwd_full, false) );
        checkCUFFT( cufftSetAutoAllocation(fwd_part, false) );
        checkCUFFT( cufftSetAutoAllocation(bwd_full, false) );
        checkCUFFT( cufftSetAutoAllocation(bwd_part, false) );

        long_t memory_per_1k_transforms = 16 * clen * 1024;
        long_t max_memory = 128 * 1024 * 1024;

        long_t how_many_1k_transforms = max_memory / memory_per_1k_transforms;
        if ( how_many_1k_transforms == 0 ) how_many_1k_transforms = 1;

        long_t how_many_at_once = how_many_1k_transforms * 1024;
        how_many_at_once = std::min(how_many_at_once, n);

        n_full = n / how_many_at_once;
        has_partial = n % how_many_at_once;

        checkCUFFT( cufftMakePlanMany(fwd_full, 1, dims, NULL, 0,
                                      clen*2, NULL, 0,
                                      clen, CUFFT_R2C,
                                      how_many_at_once, &ws) );

        workspace_size_ = std::max(workspace_size_,ws);

        checkCUFFT( cufftMakePlanMany(bwd_full, 1, dims, NULL, 0,
                                      clen, NULL, 0,
                                      clen*2, CUFFT_C2R,
                                      how_many_at_once, &ws) );

        workspace_size_ = std::max(workspace_size_,ws);

        if ( has_partial )
        {
            checkCUFFT( cufftMakePlanMany(fwd_part, 1, dims, NULL, 0,
                                          clen*2, NULL, 0,
                                          clen, CUFFT_R2C,
                                          has_partial, &ws) );

            workspace_size_ = std::max(workspace_size_,ws);

            checkCUFFT( cufftMakePlanMany(bwd_part, 1, dims, NULL, 0,
                                          clen, NULL, 0,
                                          clen*2, CUFFT_C2R,
                                          has_partial, &ws) );

            workspace_size_ = std::max(workspace_size_,ws);
        }

        delta = clen * how_many_at_once;
    }

    void forward( float* in_out, void* ws ) const
    {
        checkCUFFT(cufftSetWorkArea(fwd_full,ws));
        for ( long_t i = 0; i < n_full; ++i )
        {
            checkCUFFT(cufftExecR2C(fwd_full,in_out,
                                    reinterpret_cast<cuComplex*>(in_out)));
            in_out += delta * 2;
        }

        if ( has_partial )
        {
            checkCUFFT(cufftSetWorkArea(fwd_part,ws));
            checkCUFFT(cufftExecR2C(fwd_part,in_out,
                                    reinterpret_cast<cuComplex*>(in_out)));

        }
    }


    void backward( cuComplex* in_out, void* ws ) const
    {
        checkCUFFT(cufftSetWorkArea(bwd_full,ws));
        for ( long_t i = 0; i < n_full; ++i )
        {
            checkCUFFT(cufftExecC2R(bwd_full,in_out,
                                    reinterpret_cast<float*>(in_out)));
            in_out += delta;
        }

        if ( has_partial )
        {
            checkCUFFT(cufftSetWorkArea(bwd_part,ws));
            checkCUFFT(cufftExecC2R(bwd_part,in_out,
                                    reinterpret_cast<float*>(in_out)));
        }
    }

};


class cufft_1d_c2c_transformer
{
private:
    cufftHandle full;
    cufftHandle part;

    long_t n_full     ;
    long_t has_partial;

    long_t delta      ;
    size_t workspace_size_ = 0;

public:
    long_t workspace_size() const
    {
        return static_cast<long_t>(workspace_size_);
    }

    ~cufft_1d_c2c_transformer()
    {
        checkCUFFT( cufftDestroy(full) );
        checkCUFFT( cufftDestroy(part) );
    }

    cufft_1d_c2c_transformer( long_t l, long_t n )
    {
        int   dims[1] = { static_cast<int>(l) };
        size_t ws;

        checkCUFFT( cufftCreate(&full) );
        checkCUFFT( cufftCreate(&part) );

        checkCUFFT( cufftSetAutoAllocation(full, false) );
        checkCUFFT( cufftSetAutoAllocation(part, false) );

        long_t memory_per_1k_transforms = 16 * l * 1024;
        long_t max_memory = 128 * 1024 * 1024;

        long_t how_many_1k_transforms = max_memory / memory_per_1k_transforms;
        if ( how_many_1k_transforms == 0 ) how_many_1k_transforms = 1;

        long_t how_many_at_once = how_many_1k_transforms * 1024;
        how_many_at_once = std::min(how_many_at_once, n);

        n_full = n / how_many_at_once;
        has_partial = n % how_many_at_once;

        checkCUFFT( cufftMakePlanMany(full, 1, dims, NULL, 0,
                                      l, NULL, 0,
                                      l, CUFFT_C2C,
                                      how_many_at_once, &ws) );

        workspace_size_ = std::max(workspace_size_,ws);

        if ( has_partial )
        {
            checkCUFFT( cufftMakePlanMany(part, 1, dims, NULL, 0,
                                          l, NULL, 0,
                                          l, CUFFT_C2C,
                                          has_partial, &ws) );

            workspace_size_ = std::max(workspace_size_,ws);
        }

        delta = l * how_many_at_once;
    }

    void forward( cuComplex* in_out, void* ws ) const
    {
        checkCUFFT(cufftSetWorkArea(full,ws));
        for ( long_t i = 0; i < n_full; ++i )
        {
            checkCUFFT(cufftExecC2C(full,in_out,in_out,CUFFT_FORWARD));
            in_out += delta;
        }

        if ( has_partial )
        {
            checkCUFFT(cufftSetWorkArea(part,ws));
            checkCUFFT(cufftExecC2C(part,in_out,in_out,CUFFT_FORWARD));
        }
    }


    void backward( cuComplex* in_out, void* ws ) const
    {
        checkCUFFT(cufftSetWorkArea(full,ws));
        for ( long_t i = 0; i < n_full; ++i )
        {
            checkCUFFT(cufftExecC2C(full,in_out,in_out,CUFFT_INVERSE));
            in_out += delta;
        }

        if ( has_partial )
        {
            checkCUFFT(cufftSetWorkArea(part,ws));
            checkCUFFT(cufftExecC2C(part,in_out,in_out,CUFFT_INVERSE));
        }
    }

};


}}} // namespace znn::fwd::device
