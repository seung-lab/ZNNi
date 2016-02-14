#pragma once

#include <cufft.h>
#include "../../../types.hpp"
#include "../../../assert.hpp"
#include "../../utils.hpp"
#include "../../memory.hpp"
#include "utils.hpp"

namespace znn { namespace fwd { namespace gpu {


class cufft_padded_pruned_forward_transformer
{
private:
    cufftHandle along_z;
    cufftHandle along_y;
    cufftHandle along_x;

    vec3i  ks, os, cs;
    long_t n;

public:
    ~cufft_padded_pruned_forward_transformer()
    {
        checkCUFFT( cufftDestroy(along_x) );
        checkCUFFT( cufftDestroy(along_y) );
        checkCUFFT( cufftDestroy(along_z) );
    }

    cufft_padded_pruned_forward_transformer( vec3i const & _ks,
                                             vec3i const & _os,
                                             long_t _n )
        : ks(_ks), os(_os), cs(_os), n(_n)
    {
        cs[2] /= 2; cs[2] += 1;


        int dims[1] = { static_cast<int>(os[2]) };

        checkCUFFT( cufftPlanMany(&along_z, 1, dims, NULL, 0,
                                  cs[2]*2, NULL, 0,
                                  cs[2], CUFFT_R2C,
                                  ks[0]*ks[1]*n) );

        dims[0] = cs[1];

        checkCUFFT( cufftPlanMany(&along_y, 1, dims, NULL, 0,
                                  cs[1], NULL, 0,
                                  cs[1], CUFFT_C2C,
                                  ks[0]*cs[2]*n) );

        dims[0] = cs[0];

        checkCUFFT( cufftPlanMany(&along_x, 1, dims, NULL, 0,
                                  cs[0], NULL, 0,
                                  cs[0], CUFFT_C2C,
                                  cs[1]*cs[2]*n) );
    }

public:
    void forward( float* in, cuComplex* out, cuComplex* tmp ) const
    {
        // Expand along Z direction into out
        stage_1_scatter( ks[2], cs[2]*2, in,
                         reinterpret_cast<float*>(out),
                         ks[0]*ks[1]*ks[2]*n );

        checkCUFFT(cufftExecR2C(along_z,reinterpret_cast<float*>(out),out));

        // Expand along Y direction out->tmp
        stage_2_scatter( cs[2], ks[1], cs[1], out, tmp, cs[2]*ks[1]*ks[0]*n );

        checkCUFFT(cufftExecC2C(along_y,tmp,tmp,CUFFT_FORWARD));

        // Expand along X direction tmp->out
        stage_2_scatter( cs[1] * cs[2], ks[0], cs[0], tmp, out, cs[2]*cs[1]*ks[0]*n );

        checkCUFFT(cufftExecC2C(along_x,out,out,CUFFT_FORWARD));
    }


};

class cufft_padded_pruned_backward_transformer
{
private:
    cufftHandle along_z;
    cufftHandle along_y;
    cufftHandle along_x;

    vec3i  ks, off, os, cs;
    long_t n;

public:
    ~cufft_padded_pruned_backward_transformer()
    {
        checkCUFFT( cufftDestroy(along_x) );
        checkCUFFT( cufftDestroy(along_y) );
        checkCUFFT( cufftDestroy(along_z) );
    }

    cufft_padded_pruned_backward_transformer( vec3i const & _ks,
                                              vec3i const & _off,
                                              vec3i const & _os,
                                              long_t _n )
        : ks(_ks), off(_off), os(_os), cs(_os), n(_n)
    {
        cs[2] /= 2; cs[2] += 1;


        int dims[1] = { static_cast<int>(os[2]) };

        checkCUFFT( cufftPlanMany(&along_z, 1, dims, NULL, 0,
                                  cs[2], NULL, 0,
                                  cs[2]*2, CUFFT_C2R,
                                  ks[0]*ks[1]*n) );

        dims[0] = cs[1];

        checkCUFFT( cufftPlanMany(&along_y, 1, dims, NULL, 0,
                                  cs[1], NULL, 0,
                                  cs[1], CUFFT_C2C,
                                  ks[0]*cs[2]*n) );

        dims[0] = cs[0];

        checkCUFFT( cufftPlanMany(&along_x, 1, dims, NULL, 0,
                                  cs[0], NULL, 0,
                                  cs[0], CUFFT_C2C,
                                  cs[1]*cs[2]*n) );
    }

public:
    void backward( cuComplex* out, float* in, cuComplex* tmp ) const
    {
        checkCUFFT(cufftExecC2C(along_x,out,out,CUFFT_INVERSE));

        // Gather along X direction out->tmp
        stage_2_gather( cs[1] * cs[2], ks[0], cs[0], tmp,
                        out + off[0], cs[2]*cs[1]*ks[0]*n );

        checkCUFFT(cufftExecC2C(along_y,tmp,tmp,CUFFT_INVERSE));

        // Gather along Y direction out->tmp
        stage_2_gather( cs[2], ks[1], cs[1], out,
                        tmp + off[1], cs[2]*ks[1]*ks[0]*n );

        checkCUFFT(cufftExecC2R(along_z,out,reinterpret_cast<float*>(out)));

        // Gather along Z direction into in
        stage_1_gather( ks[2], cs[2]*2, in,
                        reinterpret_cast<float*>(out) + off[2],
                        ks[0]*ks[1]*ks[2]*n );

        div_all_by( in, in + ks[0]*ks[1]*ks[2]*n, os[0]*os[1]*os[2]);
    }


};


}}} // namespace znn::fwd::gpu
