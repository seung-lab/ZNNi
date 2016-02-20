#pragma once

#include <cufft.h>
#include "../../../types.hpp"
#include "../../../assert.hpp"
#include "../../utils.hpp"
#include "../../memory.hpp"
#include "utils.hpp"
#include "1d_transforms.hpp"

namespace znn { namespace fwd { namespace gpu {


class cufft_padded_pruned_forward_transformer
{
private:
    vec3i  ks, os, cs;
    long_t n;

    cufft_1d_r2c_transformer along_z;
    cufft_1d_c2c_transformer along_y;
    cufft_1d_c2c_transformer along_x;

    size_t workspace_size_ = 0;

public:
    long_t workspace_size() const
    {
        return workspace_size_;
    }

    cufft_padded_pruned_forward_transformer( vec3i const & _ks,
                                             vec3i const & _os,
                                             long_t _n )
        : ks(_ks)
        , os(_os)
        , cs(os[0],os[1],os[2]/2+1)
        , n(_n)
        , along_z(os[2],ks[0]*ks[1]*n)
        , along_y(cs[1],ks[0]*cs[2]*n)
        , along_x(cs[0],cs[1]*cs[2]*n)
    {
        workspace_size_ = std::max({along_z.workspace_size(),
                    along_y.workspace_size(),
                    along_x.workspace_size()});
    }

public:
    void forward( float* in, cuComplex* out, cuComplex* tmp, void* ws ) const
    {
        // Expand along Z direction into out
        stage_1_scatter( ks[2], cs[2]*2, in,
                         reinterpret_cast<float*>(out),
                         ks[0]*ks[1]*ks[2]*n );
        along_z.forward(reinterpret_cast<float*>(out),ws);

        // Expand along Y direction out->tmp
        stage_2_scatter( cs[2], ks[1], cs[1], out, tmp, cs[2]*ks[1]*ks[0]*n );
        along_y.forward(tmp,ws);

        // Expand along X direction tmp->out
        stage_2_scatter( cs[1] * cs[2], ks[0], cs[0], tmp, out, cs[2]*cs[1]*ks[0]*n );
        along_x.forward(out,ws);
    }


};

class cufft_padded_pruned_backward_transformer
{
private:
    vec3i  ks, off, os, cs;
    long_t n;

    cufft_1d_r2c_transformer along_z;
    cufft_1d_c2c_transformer along_y;
    cufft_1d_c2c_transformer along_x;

    size_t workspace_size_ = 0;

public:
    long_t workspace_size() const
    {
        return workspace_size_;
    }

    cufft_padded_pruned_backward_transformer( vec3i const & _ks,
                                              vec3i const & _off,
                                              vec3i const & _os,
                                              long_t _n )
        : ks(_ks)
        , off(_off)
        , os(_os)
        , cs(os[0],os[1],os[2]/2+1)
        , n(_n)
        , along_z(os[2],ks[0]*ks[1]*n)
        , along_y(cs[1],ks[0]*cs[2]*n)
        , along_x(cs[0],cs[1]*cs[2]*n)
    {
        workspace_size_ = std::max({along_z.workspace_size(),
                    along_y.workspace_size(),
                    along_x.workspace_size()});
    }

public:
    void backward( cuComplex* out, float* in, cuComplex* tmp, void* ws ) const
    {
        along_x.backward(out,ws);

        // Gather along X direction out->tmp
        stage_2_gather( cs[1] * cs[2], ks[0], cs[0], tmp,
                        out + off[0], cs[2]*cs[1]*ks[0]*n );

        along_y.backward(tmp,ws);

        // Gather along Y direction out->tmp
        stage_2_gather( cs[2], ks[1], cs[1], out,
                        tmp + off[1], cs[2]*ks[1]*ks[0]*n );

        along_z.backward(out,ws);

        // Gather along Z direction into in
        stage_1_gather( ks[2], cs[2]*2, in,
                        reinterpret_cast<float*>(out) + off[2],
                        ks[0]*ks[1]*ks[2]*n );

        div_all_by( in, in + ks[0]*ks[1]*ks[2]*n, os[0]*os[1]*os[2]);
    }


};


}}} // namespace znn::fwd::gpu
