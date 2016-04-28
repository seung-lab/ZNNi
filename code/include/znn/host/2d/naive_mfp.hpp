#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/2d/host_layer.hpp"

namespace znn { namespace fwd { namespace host { namespace twod {

class naive_mfp2d
    : public mfp_layer2d<host_layer2d>
{
public:
    naive_mfp2d( long_t n, long_t finout,
                 vec2i const & is, vec2i const & ws ) noexcept
        : mfp_layer2d<host_layer2d>(n,finout,is,ws)
    { }

    host_tensor<real,4> forward( host_tensor<real,4> in ) const override
    {
        host_tensor<float,4> ret(output_shape);
        host_tensor_ref<float,5> out(ret.data(),
                                     in_batch_size,
                                     num_fragments,
                                     num_inputs,
                                     out_image_size[0],
                                     out_image_size[1]);

        vec2i ws = window_size;
        vec2i os = out_image_size;

        long_t w = 0;
        for ( long_t xoff = 0; xoff < ws[0]; ++xoff )
            for ( long_t yoff = 0; yoff < ws[1]; ++yoff, ++w )
                for ( long_t b = 0; b < in_batch_size; ++b )
                    for ( long_t l = 0; l < num_inputs; ++l )
                        for ( long_t xo = 0; xo < os[0]; ++xo )
                            for ( long_t yo = 0; yo < os[1]; ++yo )
                            {
                                out[b][w][l][xo][yo] = std::numeric_limits<real>::lowest();
                                for ( long_t x = 0; x < ws[0]; ++x )
                                    for ( long_t y = 0; y < ws[1]; ++y )
                                        out[b][w][l][xo][yo]
                                            = std::max(out[b][w][l][xo][yo],
                                                       in[b][l][ws[0]*xo+xoff+x][ws[1]*yo+yoff+y]);
                                    }

        return ret;
    }

    long_t resident_memory() const override
    {
        return 0;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

};


}}}} // namespace znn::fwd::host::twod
