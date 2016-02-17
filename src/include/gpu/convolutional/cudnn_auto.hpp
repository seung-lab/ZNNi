#pragma once

#include "cudnn.hpp"
#include "cudnn_single_batch.hpp"
#include "cudnn_single_output.hpp"

namespace znn { namespace fwd { namespace gpu {


class cudnn_auto_convolutional_layer
    : public convolutional_layer_base
    , public device_layer
{
private:
    std::unique_ptr<device_layer> layer_;

public:
    device_array<float> forward( device_array<float> in ) const override
    {
        return layer_->forward(std::move(in));
    }

public:
    cudnn_auto_convolutional_layer( handle_t& handle,
                                    long_t n, long_t fin, long_t fout,
                                    vec3i const & is, vec3i const & ks,
                                    float* km = nullptr, float* bs = nullptr )
        : convolutional_layer_base(n,fin,fout,is,ks)
    {
        if ( fin * fout * out_image_len > 250000000 )
        {
            layer_ = std::unique_ptr<device_layer>
                ( new cudnn_single_output_convolutional_layer
                  (handle, n, fin, fout,
                   is, ks, km, bs));
        }
        else if ( fin * fout * n * out_image_len > 250000000 )
        {
            layer_ = std::unique_ptr<device_layer>
                ( new cudnn_single_batch_convolutional_layer
                  (handle, n, fin, fout,
                   is, ks, km, bs));
        }
        else
        {
            layer_ = std::unique_ptr<device_layer>
                ( new cudnn_convolutional_layer(handle, n, fin, fout,
                                                is, ks, km, bs));
        }
    }
};



}}} // namespace znn::fwd::gpu
