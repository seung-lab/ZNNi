#pragma once

#include "../../layer.hpp"
#include <boost/multi_array.hpp>

#include <limits>

namespace znn { namespace fwd { namespace cpu {

class naive_pooling_layer
    : public pooling_layer_base
    , public host_layer
{
public:
    naive_pooling_layer( long_t n, long_t fin,
                         vec3i const & is, vec3i const & ks )
        : pooling_layer_base( n, fin, is, ks )
    { }

    host_array<real> forward( host_array<real> m ) const override
    {
        host_array<real> ret = get_array<real>(total_output_len);

        boost::multi_array_ref<real,5> in(m.get(),
                                          boost::extents
                                          [in_batch_size]
                                          [num_inputs]
                                          [in_image_size[0]]
                                          [in_image_size[1]]
                                          [in_image_size[2]]);

        boost::multi_array_ref<real,6> out(ret.get(),
                                           boost::extents
                                           [in_batch_size]
                                           [window_len]
                                           [num_outputs]
                                           [out_image_size[0]]
                                           [out_image_size[1]]
                                           [out_image_size[2]]);


        vec3i ws = window_size;
        vec3i os = out_image_size;

        long_t w = 0;
        for ( long_t xoff = 0; xoff < ws[0]; ++xoff )
            for ( long_t yoff = 0; yoff < ws[1]; ++yoff )
                for ( long_t zoff = 0; zoff < ws[2]; ++zoff, ++w )
                    for ( long_t b = 0; b < in_batch_size; ++b )
                        for ( long_t l = 0; l < num_inputs; ++l )
                            for ( long_t xo = 0; xo < os[0]; ++xo )
                                for ( long_t yo = 0; yo < os[1]; ++yo )
                                    for ( long_t zo = 0; zo < os[2]; ++zo )
                                    {
                                        out[b][w][l][xo][yo][zo] = std::numeric_limits<real>::lowest();
                                        for ( long_t x = 0; x < ws[0]; ++x )
                                            for ( long_t y = 0; y < ws[1]; ++y )
                                                for ( long_t z = 0; z < ws[2]; ++z )
                                                    out[b][w][l][xo][yo][zo]
                                                        = std::max(out[b][w][l][xo][yo][zo],
                                                                   in[b][l][ws[0]*xo+xoff+x][ws[1]*yo+yoff+y][ws[2]*zo+zoff+z]);
                                    }

        return ret;
    }
};

}}} // namespace znn::fwd::cpu
