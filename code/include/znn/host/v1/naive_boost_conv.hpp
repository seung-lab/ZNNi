#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/host_layer.hpp"
#include "znn/host/v1/conv_data.hpp"

#include <boost/multi_array.hpp>

namespace znn { namespace fwd { namespace host { namespace v1 {

class naive_boost_conv
    : public conv_layer<host_layer>
    , public conv_data
{
public:
    naive_boost_conv( long_t n, long_t fin, long_t fout,
                      vec3i const & is, vec3i const & ks,
                      float * km = nullptr, float* bs = nullptr )
        : conv_layer<host_layer>(n,fin,fout,is,ks)
        , conv_data(fin,fout,ks,km,bs)
    { }


    host_tensor<float,5> forward( host_tensor<float,5> m ) override
    {
        host_tensor<float,5> ret(output_shape);

        boost::multi_array_ref<real,5> in(m.data(),
                                          boost::extents
                                          [in_batch_size]
                                          [num_inputs]
                                          [in_image_size[0]]
                                          [in_image_size[1]]
                                          [in_image_size[2]]);

        boost::multi_array_ref<real,5> ks(conv_data::kernels.data(),
                                          boost::extents
                                          [num_outputs]
                                          [num_inputs]
                                          [kernel_size[0]]
                                          [kernel_size[1]]
                                          [kernel_size[2]]);

        boost::multi_array_ref<real,5> out(ret.data(),
                                           boost::extents
                                           [out_batch_size]
                                           [num_outputs]
                                           [out_image_size[0]]
                                           [out_image_size[1]]
                                           [out_image_size[2]]);


        real* bs = conv_data::biases.data();
        vec3i kl = kernel_size;

        for ( long_t b = 0; b < in_batch_size; ++b )
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

    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

};


}}}} // namespace znn::fwd::host::v1
