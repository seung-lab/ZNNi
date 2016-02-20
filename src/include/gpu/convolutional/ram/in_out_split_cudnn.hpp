#pragma once

#include "in_split_cudnn.hpp"

namespace znn { namespace fwd { namespace gpu {

template<typename Native>
class in_out_split_cudnn_convolutional_layer
    : public convolutional_layer_base
{
private:
    std::unique_ptr<in_split_cudnn_convolutional_layer<Native>> full_   ;
    std::unique_ptr<in_split_cudnn_convolutional_layer<Native>> partial_;

    long_t n_full_;

    long_t workspace_size_ = 0;

public:
    long_t workspace_size() const
    {
        return workspace_size_;
    }

    void forward(float* in,
                 float* out,
                 float* kernels,
                 float* biases,
                 void* workspace_d) const
    {
        for ( long_t i = 0; i < n_full_; ++i )
        {
            full_->forward(in, out, kernels, biases, workspace_d);

            out     += full_->output_len ;
            kernels += full_->kernels_len;
            biases  += full_->num_outputs;
        }

        if ( partial_ )
        {
            partial_->forward(in, out, kernels, biases, workspace_d);
        }
    }

public:
    in_out_split_cudnn_convolutional_layer( long_t fin,
                                            long_t fin_chunk,
                                            long_t fout,
                                            long_t fout_chunk,
                                            vec3i const & is,
                                            vec3i const & ks )
        : convolutional_layer_base(1, fin, fout, is, ks)
    {
        n_full_ = fout / fout_chunk;

        full_ = std::unique_ptr<in_split_cudnn_convolutional_layer<Native>>
            ( new in_split_cudnn_convolutional_layer<Native>
              (fin, fin_chunk, fout_chunk, is, ks));

        workspace_size_ = full_->workspace_size();

        if ( fout % fout_chunk )
        {
            partial_ = std::unique_ptr<in_split_cudnn_convolutional_layer<Native>>
                ( new in_split_cudnn_convolutional_layer<Native>
                  (fin, fin_chunk, fout%fout_chunk, is, ks));

            workspace_size_ = std::max(workspace_size_,
                                       partial_->workspace_size());
        }
    }
};

}}} // namespace znn::fwd::gpu
