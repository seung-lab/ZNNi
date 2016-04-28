#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/2d/host_layer.hpp"
#include "znn/host/2d/conv_data.hpp"

#include <boost/multi_array.hpp>

namespace znn { namespace fwd { namespace host { namespace twod {

class naive_boost_conv2d
    : public conv_layer2d<host_layer2d>
    , public conv_data2d
{
public:
    naive_boost_conv2d( long_t n, long_t fin, long_t fout,
                        vec2i const & is, vec2i const & ks,
                        float * km = nullptr, float* bs = nullptr )
        : conv_layer2d<host_layer2d>(n,fin,fout,is,ks)
        , conv_data2d(fin,fout,ks,km,bs)
    { }


    host_tensor<float,4> forward( host_tensor<float,4> m ) const override
    {
        host_tensor<float,4> ret(output_shape);

        boost::multi_array_ref<real,4> in(m.data(),
                                          boost::extents
                                          [in_batch_size]
                                          [num_inputs]
                                          [in_image_size[0]]
                                          [in_image_size[1]]);

        boost::const_multi_array_ref<real,4> ks(conv_data2d::kernels.data(),
                                                boost::extents
                                                [num_outputs]
                                                [num_inputs]
                                                [kernel_size[0]]
                                                [kernel_size[1]]);

        boost::multi_array_ref<real,4> out(ret.data(),
                                           boost::extents
                                           [out_batch_size]
                                           [num_outputs]
                                           [out_image_size[0]]
                                           [out_image_size[1]]);


        real const * bs = conv_data2d::biases.data();
        vec2i kl = kernel_size;

        for ( long_t b = 0; b < in_batch_size; ++b )
            for ( long_t oi = 0; oi < num_outputs; ++oi )
                for ( long_t ox = 0; ox < out_image_size[0]; ++ox )
                    for ( long_t oy = 0; oy < out_image_size[1]; ++oy )
                    {
                        out[b][oi][ox][oy] = bs[oi];

                        for ( long_t ii = 0; ii < num_inputs; ++ii )
                            for ( long_t kx = 0; kx < kernel_size[0]; ++kx )
                                for ( long_t ky = 0; ky < kernel_size[1]; ++ky )
                                    out[b][oi][ox][oy] +=
                                        in[b][ii][ox+kx][oy+ky] *
                                        ks[oi][ii][kl[0]-kx-1][kl[1]-ky-1];
                    }

        return ret;
   }

    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

};


}}}} // namespace znn::fwd::host::twod
