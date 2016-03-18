#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"
#include "znn/tensor/tensor.hpp"
#include <boost/multi_array.hpp>

namespace znn { namespace fwd {

class deshuffler
{
private:
    vec3i  original_size;
    vec3i  current_size ;
    long_t n            ;
    long_t len          ;

    host_array<uint32_t> mappings;

private:
    void single_shuffle( uint32_t* inp, uint32_t* outp,
                         vec3i const & is, vec3i const & ss ) noexcept
    {
        boost::multi_array_ref<uint32_t,3>
            in(inp, boost::extents[is[0]][is[1]][is[2]]);

        vec3i os = is / ss;
        long_t olen = os[0]*os[1]*os[2];

        for ( long_t ox = 0; ox < ss[0]; ++ox )
            for ( long_t oy = 0; oy < ss[1]; ++oy )
                for ( long_t oz = 0; oz < ss[2]; ++oz )
                {
                    boost::multi_array_ref<uint32_t,3>
                        out(outp, boost::extents[os[0]][os[1]][os[2]]);

                    for ( long_t x = 0; x < os[0]; ++x )
                        for ( long_t y = 0; y < os[1]; ++y )
                            for ( long_t z = 0; z < os[2]; ++z )
                                out[x][y][z]
                                    = in[ox+x*ss[0]][oy+y*ss[1]][oz+z*ss[2]];

                    outp += olen;
                }
    }

public:

    explicit deshuffler( vec3i const & s )
        : original_size(s)
        , current_size(s)
        , n(1)
        , len(s[0]*s[1]*s[2])
        , mappings(len)
    {
        for ( long_t i = 0; i < len; ++i )
        {
            mappings.data()[i] = static_cast<uint32_t>(i);
        }
    }

    void split( vec3i const & s )
    {
        STRONG_ASSERT( current_size % s == vec3i::zero );

        host_array<uint32_t> new_mappings(len);

        long_t delta = current_size[0] * current_size[1] * current_size[2];

        for ( long_t i = 0; i < n; ++i )
        {
            single_shuffle( mappings.data() + i * delta,
                            new_mappings.data() + i * delta,
                            current_size,
                            s );
        }

        mappings = std::move(new_mappings);

        n *= s[0]*s[1]*s[2];
        current_size /= s;
    }

    host_array<real> shuffle(host_array<real> in) const
    {
        host_array<real> ret(len);

        for ( long_t i = 0; i < len; ++i )
        {
            ret.data()[i] = in.data()[mappings.data()[i]];
        }

        return ret;
    }

    host_array<real> deshuffle(host_array<real> in) const
    {
        auto ret = get_array<real>(len);

        for ( long_t i = 0; i < len; ++i )
        {
            ret.data()[mappings.data()[i]] = in.data()[i];
        }

        return ret;
    }

};

}} // namespace znn::fwd
