#pragma once

#include "../../layer.hpp"
#include "../host_layer.hpp"
#include "base.hpp"

#include <boost/multi_array.hpp>

namespace znn { namespace fwd { namespace cpu {

class naive_convolutional_layer
    : public cpu_convolutional_layer_base
    , public host_layer
{
public:
    naive_convolutional_layer( long_t n, long_t fin, long_t fout,
                               vec3i const & is, vec3i const & ks,
                               real * km = nullptr, real* bs = nullptr )
        : cpu_convolutional_layer_base( n, fin, fout, is, ks, km, bs )
    { }

    host_array<real> forward( host_array<real> m ) const override
    {
        host_array<real> ret = get_array<real>(total_output_len);

        boost::multi_array_ref<real,5> in(m.get(),
                                          boost::extents
                                          [batch_size]
                                          [num_inputs]
                                          [in_image_size[0]]
                                          [in_image_size[1]]
                                          [in_image_size[2]]);

        boost::multi_array_ref<real,5> ks(get_kernels(),
                                          boost::extents
                                          [num_outputs]
                                          [num_inputs]
                                          [kernel_size[0]]
                                          [kernel_size[1]]
                                          [kernel_size[2]]);

        boost::multi_array_ref<real,5> out(ret.get(),
                                           boost::extents
                                           [batch_size]
                                           [num_outputs]
                                           [out_image_size[0]]
                                           [out_image_size[1]]
                                           [out_image_size[2]]);


        real* bs = get_biases();
        vec3i kl = kernel_size;

        for ( long_t b = 0; b < batch_size; ++b )
            for ( long_t oi = 0; oi < num_outputs; ++oi )
                for ( long_t ox = 0; ox < out_image_size[0]; ++ox )
                    for ( long_t oy = 0; oy < out_image_size[1]; ++oy )
                        for ( long_t oz = 0; oz < out_image_size[2]; ++oz )
                        {
                            out[b][oi][ox][oy][oz] = bs[oi];

                            for ( long_t ii = 0; ii < num_inputs; ++ii )
                                for ( long_t kx = 0; kx < kernel_size[0]; ++kx )
                                    for ( long_t ky = 0; ky < kernel_size[1]; ++ky )
                                        for ( long_t kz = 0; kz < kernel_size[2]; ++kz )

                                            out[b][oi][ox][oy][oz] +=
                                                in[b][ii][ox+kx][oy+ky][oz+kz] *
                                                ks[oi][ii][kl[0]-kx-1][kl[1]-ky-1][kl[2]-kz-1];
                        }

        return ret;
    }
};

}}} // namespace znn::fwd::cpu
