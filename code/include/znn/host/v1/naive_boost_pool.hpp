#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/host_layer.hpp"

#include <boost/multi_array.hpp>

namespace znn { namespace fwd { namespace host { namespace v1 {

class naive_boost_pool
    : public pool_layer<host_layer>
{
public:
    naive_boost_pool( long_t n, long_t finout,
                      vec3i const & is, vec3i const & ws ) noexcept
        : pool_layer<host_layer>(n,finout,is,ws)
    { }

    host_tensor<real,5> forward( host_tensor<real,5> m ) const override
    {
        host_tensor<float,5> ret(output_shape);

        boost::multi_array_ref<real,5> in(m.data(),
                                          boost::extents
                                          [in_batch_size]
                                          [num_inputs]
                                          [in_image_size[0]]
                                          [in_image_size[1]]
                                          [in_image_size[2]]);

        boost::multi_array_ref<real,5> out(ret.data(),
                                           boost::extents
                                           [in_batch_size]
                                           [num_outputs]
                                           [out_image_size[0]]
                                           [out_image_size[1]]
                                           [out_image_size[2]]);


        vec3i ws = window_size;
        vec3i os = out_image_size;

        for ( long_t b = 0; b < in_batch_size; ++b )
            for ( long_t l = 0; l < num_inputs; ++l )
                for ( long_t xo = 0; xo < os[0]; ++xo )
                    for ( long_t yo = 0; yo < os[1]; ++yo )
                        for ( long_t zo = 0; zo < os[2]; ++zo )
                        {
                            out[b][l][xo][yo][zo] = std::numeric_limits<real>::lowest();
                            for ( long_t x = 0; x < ws[0]; ++x )
                                for ( long_t y = 0; y < ws[1]; ++y )
                                    for ( long_t z = 0; z < ws[2]; ++z )
                                        out[b][l][xo][yo][zo]
                                            = std::max(out[b][l][xo][yo][zo],
                                                       in[b][l][xo*ws[0]+x][yo*ws[1]+y][zo*ws[2]+z]);
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


}}}} // namespace znn::fwd::host::v1
