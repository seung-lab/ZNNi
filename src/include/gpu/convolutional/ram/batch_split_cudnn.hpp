#pragma once

#include "native_cudnn.hpp"
#include "base.hpp"
#include "../../../layer.hpp"

namespace znn { namespace fwd { namespace gpu {

template<typename Native>
class batch_split_cudnn_convolutional_layer
    : public convolutional_layer_base
{
private:
    std::unique_ptr<Native> full_   ;
    std::unique_ptr<Native> partial_;

    long_t n_full_;

    long_t workspace_size_ = 0;

public:
    long_t workspace_size() const override
    {
        return workspace_size_;
    }

    void forward(float* in,
                 float* out,
                 float* kernels,
                 float* biases,
                 void* workspace_d) const override
    {
        auto in_d      = get_device_array<float>(full_->total_input_len);
        auto out_d     = get_device_array<float>(full_->total_output_len);
        auto kernels_d = get_device_array<float>(kernels_len);
        auto biases_d  = get_device_array<float>(num_outputs);

        // copy the biases and kernels
        device_copy_n( biases, num_outputs, biases_d );
        device_copy_n( kernels, kernels_len, kernels_d );

        for ( long_t i = 0; i < n_full_; ++i )
        {
            // copy the input
            device_copy_n( in, full_->total_input_len, in_d );

            full_->forward(in_d.get(), out_d.get(),
                           kernels_d.get(), 0, workspace_d);

            full_->apply_bias(out_d.get(), biases_d.get());

            checkCudaErrors( cudaMemcpy(out, out_d.get(),
                                        full_->total_output_len * sizeof(float),
                                        cudaMemcpyDeviceToHost) );
            in      += full_->total_input_len;
            out     += full_->total_output_len;
        }

        if ( partial_ )
        {
            // copy the input
            device_copy_n( in, partial_->total_input_len, in_d );

            partial_->forward(in_d.get(), out_d.get(),
                              kernels_d.get(), 0, workspace_d);

            partial_->apply_bias(out_d.get(), biases_d.get());

            checkCudaErrors( cudaMemcpy(out, out_d.get(),
                                        partial_->total_output_len
                                        * sizeof(float),
                                        cudaMemcpyDeviceToHost) );
        }
    }

public:
    batch_split_cudnn_convolutional_layer( long_t n,
                                           long_t n_chunk,
                                           long_t fin,
                                           long_t fout,
                                           vec3i const & is,
                                           vec3i const & ks )
        : convolutional_layer_base(n, fin, fout, is, ks)
    {
        n_full_ = n / n_chunk;

        full_ = std::unique_ptr<Native>
            ( new Native(n_chunk, fin, fout, is, ks));

        workspace_size_ = full_->workspace_size();

        if ( n % n_chunk )
        {
            partial_ = std::unique_ptr<Native>
                ( new Native(n%n_chunk, fin, fout, is, ks));

            workspace_size_ = std::max(workspace_size_,
                                       partial_->workspace_size());
        }
    }

};

}}} // namespace znn::fwd::gpu
