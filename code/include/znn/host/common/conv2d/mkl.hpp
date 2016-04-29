#pragma once

#include "znn/assert.hpp"
#include "znn/types.hpp"

#include <mkl_vsl.h>

namespace znn { namespace fwd { namespace host {

class convolver2d
{
private:
    VSLConvTaskPtr conv_;

public:
    ~convolver2d()
    {
        vlsConvDeleteTask(&conv_);
    }

    convolver2d( vec2i const & i, vec2i const & k )
        : ix(i)
        , kx(k)
        , rx(i - k + vec2i::one)
    {

        int status;
        const MKL_INT start[2]={k[1]-1,k[0]-1};

        MKL_INT shape[3]  = { i[1], i[0] };
        MKL_INT bshape[3] = { k[1], k[0] };
        MKL_INT rshape[3] = { i[1] + 1 - k[1],
                              i[0] + 1 - k[0] };

#if defined(ZNN_USE_LONG_DOUBLE_PRECISION)

#elif defined(ZNN_USE_DOUBLE_PRECISION)
        status = vsldConvNewTask(&conv_,VSL_CONV_MODE_DIRECT,2,
                                 shape, bshape, rshape);
#else
        status = vslsConvNewTask(&conv_,VSL_CONV_MODE_DIRECT,2,
                                 shape, bshape, rshape);
#endif

        status = vslConvSetStart(conv_, start);
    }

    void convolve( real* in, real const* kernel, real* out ) const
    {
#if defined(ZNN_USE_LONG_DOUBLE_PRECISION)

#elif defined(ZNN_USE_DOUBLE_PRECISION)
        int status = vsldConvExec(conv_, in, NULL, kernel, NULL, out, NULL);
#else
        int status = vslsConvExec(conv_, in, NULL, kernel, NULL, out, NULL);
#endif
    }

};


}}} // namespace znn::fwd::host
