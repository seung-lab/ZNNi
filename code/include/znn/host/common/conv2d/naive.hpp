#pragma once

#include "znn/assert.hpp"
#include "znn/types.hpp"

namespace znn { namespace fwd { namespace host {

class convolver2d
{
private:
    class ref
    {
    private:
        long_t stride;
        real*  data;

    public:
        ref( long_t y, real* d )
            : stride(y)
            , data(d)
        {}

        real & operator()( long_t x, long_t y )
        {
            return data[x*stride + y];
        }
    };

    class cref
    {
    private:
        long_t stride;
        real const *  data;

    public:
        cref( long_t y, real const * d )
            : stride(y)
            , data(d)
        {}

        real const & operator()( long_t x, long_t y ) const
        {
            return data[x*stride + y];
        }
    };

private:
    vec2i ix, kx, rx;

public:
    convolver2d( vec2i const & i, vec2i const & k )
        : ix(i)
        , kx(k)
        , rx(i - k + vec2i::one)
    { }

    void convolve( real* in, real const* kernel, real* out ) const
    {
        ref  a(ix[1], in);
        cref b(kx[1], kernel);
        ref  r(rx[1], out);

        for ( long_t x = 0; x < rx[0]; ++x )
            for ( long_t y = 0; y < rx[1]; ++y )
            {
                real & res = r(x,y);
                res = 0;
                for ( long_t dx = x, wx = kx[0] - 1; dx < kx[0] + x; ++dx, --wx )
                    for ( long_t dy = y, wy = kx[1] - 1; dy < kx[1] + y; ++dy, --wy )
                        res += a(dx,dy) * b(wx,wy);
            }
    }

    void convolve_add( real* in, real const* kernel, real* out ) const
    {
        ref  a(ix[1], in);
        cref b(kx[1], kernel);
        ref  r(rx[1], out);

        for ( long_t x = 0; x < rx[0]; ++x )
            for ( long_t y = 0; y < rx[1]; ++y )
            {
                real & res = r(x,y);
                for ( long_t dx = x, wx = kx[0] - 1; dx < kx[0] + x; ++dx, --wx )
                    for ( long_t dy = y, wy = kx[1] - 1; dy < kx[1] + y; ++dy, --wy )
                        res += a(dx,dy) * b(wx,wy);
            }
    }
};


}}} // namespace znn::fwd::host
