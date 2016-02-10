#pragma once

#include <cudnn.h>
#include "../handle.hpp"
#include "../utils.hpp"
#include "../memory.hpp"
#include "../device_layer.hpp"
#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../layer.hpp"

namespace znn { namespace fwd { namespace gpu {

class cudnn_pooling_layer
    : public pooling_layer_base
    , public device_layer
{
private:
    handle_t&           handle_  ;
    cudnnTensorDescriptor_t  in_desc_ ;
    cudnnTensorDescriptor_t  out_desc_;

    cudnnPoolingDescriptor_t pooling_desc_;

public:
    device_array<float> forward( device_array<float> in ) const override
    {
        auto out = get_device_array<float>(total_output_len);

        long_t delta = num_inputs * out_image_len;

        float alpha = 1;
        float beta  = 0;

        vec3i is = in_image_size;

        float * out_ptr = out.get();

        for ( long_t x = 0; x < window_size[0]; ++x )
            for ( long_t y = 0; y < window_size[1]; ++y )
                for ( long_t z = 0; z < window_size[2]; ++z )
                {
                    checkCUDNN( cudnnPoolingForward(
                                    handle_.cudnn_handle,
                                    pooling_desc_,
                                    &alpha, in_desc_,
                                    in.get() + x*is[1]*is[2] + y*is[2] + z,
                                    &beta, out_desc_, out_ptr) );

                    out_ptr += delta;
                }

        return out;
    }

    ~cudnn_pooling_layer()
    {
        checkCUDNN( cudnnDestroyPoolingDescriptor(pooling_desc_) );
        checkCUDNN( cudnnDestroyTensorDescriptor(in_desc_) );
        checkCUDNN( cudnnDestroyTensorDescriptor(out_desc_) );
    }

    cudnn_pooling_layer( handle_t & handle,
                         long_t n, long_t c,
                         vec3i const & is,
                         vec3i const & ws )
        : pooling_layer_base( n, c, is, ws )
        , handle_(handle)
    {
        checkCUDNN( cudnnCreateTensorDescriptor(&in_desc_ ) );
        checkCUDNN( cudnnCreateTensorDescriptor(&out_desc_) );

        vec3i eis = out_image_size * ws;

        {
            int dims[5] = { static_cast<int>(n),
                            static_cast<int>(c),
                            static_cast<int>(eis[0]),
                            static_cast<int>(eis[1]),
                            static_cast<int>(eis[2]) };

            int strides[5] = { static_cast<int>(c*is[0]*is[1]*is[2]),
                               static_cast<int>(is[0]*is[1]*is[2]),
                               static_cast<int>(is[1]*is[2]),
                               static_cast<int>(is[2]),
                               static_cast<int>(1) };

            checkCUDNN( cudnnSetTensorNdDescriptor(
                            in_desc_,
                            CUDNN_DATA_FLOAT,
                            5, dims, strides) );
        }


        {
            vec3i os = out_image_size;

            int dims[5] = { static_cast<int>(n),
                            static_cast<int>(c),
                            static_cast<int>(os[0]),
                            static_cast<int>(os[1]),
                            static_cast<int>(os[2]) };

            int strides[5] = { static_cast<int>(c*out_image_len*window_len),
                               static_cast<int>(os[0]*os[1]*os[2]),
                               static_cast<int>(os[1]*os[2]),
                               static_cast<int>(os[2]),
                               static_cast<int>(1) };

            checkCUDNN( cudnnSetTensorNdDescriptor(
                            out_desc_,
                            CUDNN_DATA_FLOAT,
                            5, dims, strides) );
        }

        checkCUDNN( cudnnCreatePoolingDescriptor(&pooling_desc_) );

        {
            int window[3]  = { static_cast<int>(ws[0]),
                               static_cast<int>(ws[1]),
                               static_cast<int>(ws[2]) };

            int padding[3] = {0,0,0};

            checkCUDNN( cudnnSetPoolingNdDescriptor(
                            pooling_desc_,
                            CUDNN_POOLING_MAX,
                            3, window, padding, window ));
        }

    }
};


}}} // namespace znn::fwd::gpu
