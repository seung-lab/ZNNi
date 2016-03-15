#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/device_layer.hpp"
#include "znn/device/v1/conv_data.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/handle.hpp"


namespace znn { namespace fwd { namespace device { namespace v1 {

class cudnn_mfp
    : public mfp_layer<device_layer>
{
private:
    cudnnTensorDescriptor_t  in_desc_     ;
    cudnnTensorDescriptor_t  out_desc_    ;
    cudnnPoolingDescriptor_t pooling_desc_;

public:
    long_t resident_memory() const override
    {
        return 0;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

    device_tensor<float,5> forward( device_tensor<float,5> in ) const override
    {
        device_tensor<real,5> out(output_shape);

        long_t delta = num_inputs * out_image_len;

        float alpha = 1;
        float beta  = 0;

        vec3i is = in_image_size;

        float * out_ptr = out.data();

        for ( long_t x = 0; x < window_size[0]; ++x )
            for ( long_t y = 0; y < window_size[1]; ++y )
                for ( long_t z = 0; z < window_size[2]; ++z )
                {
                    checkCUDNN( cudnnPoolingForward(
                                    handle.cudnn_handle,
                                    pooling_desc_,
                                    &alpha, in_desc_,
                                    in.data() + x*is[1]*is[2] + y*is[2] + z,
                                    &beta, out_desc_, out_ptr) );

                    out_ptr += delta;
                }

        return out;
    }

    ~cudnn_mfp()
    {
        checkCUDNN( cudnnDestroyPoolingDescriptor(pooling_desc_) );
        checkCUDNN( cudnnDestroyTensorDescriptor(in_desc_) );
        checkCUDNN( cudnnDestroyTensorDescriptor(out_desc_) );
    }

    cudnn_mfp( long_t n, long_t c,
               vec3i const & is, vec3i const & ws )
        : mfp_layer<device_layer>(n,c,is,ws)
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

            int strides[5] = { static_cast<int>(c*out_image_len*num_fragments),
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


}}}} // namespace znn::fwd::device::v1
