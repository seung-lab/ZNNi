#pragma once

#include "in_out_split_cudnn.hpp"

namespace znn { namespace fwd { namespace gpu {

class batch_split_cudnn_convolutional_layer
    : public gpuram_layer_base
{
private:
    std::unique_ptr<in_out_split_cudnn_convolutional_layer> single_;
    long_t n_singles_;

public:
    long_t workspace_size() const override
    {
        return workspace_size_;
    }

    void forward(float* in,
                 float* out,
                 float* kernels,
                 float* biases,
                 float* workspace_d) const override
    {
        auto in_d      = get_device_array<float>(full_->input_len);
        auto out_d     = get_device_array<float>(full_->output_len);
        auto kernels_d = get_device_array<float>(full_->kernels_len);
        auto biases_d  = get_device_array<float>(full_->num_outputs);

        // copy the biases
        device_copy_n( biases.get(), full_->num_outputs, biases_d );

        long_t hkernel_stride = kernel_len * num_inputs;
        long_t dkernel_stride = kernel_len * full_->num_inputs;

        for ( long_t i = 0; i < n_full_; ++i )
        {
            // copy the input
            device_copy_n( in, full_->input_len, in_d );

            // copy the kernels
            if ( hkernel_stride == dkernel_stride )
            {
                device_copy_n(kernels, kernels_len, kernels_d);
            }
            else
            {
                for ( long_t k = 0; k < num_outputs; ++k )
                {
                    checkCudaErrors(
                        cudaMemcpy(kernels_d.get() + k * dkernel_stride,
                                   kernels         + k * hkernel_stride,
                                   dkernel_stride * sizeof(float),
                                   cudaMemcpyHostToDevice) );
                }
            }

            full_->forward(in_d.get(), out_d.get(),
                           kernels_d.get(), (i==0)?0:1, workspace_d);

            in      += full_->input_len;
            kernels += dkernel_stride;
        }

        if ( partial_ )
        {
            long_t pkernel_stride = kernel_len * partial_->num_inputs;

            // copy the input
            device_copy_n( in, partial_->input_len, in_d );

            // copy the kernels
            {
                for ( long_t k = 0; k < num_outputs; ++k )
                {
                    checkCudaErrors(
                        cudaMemcpy(kernels_d.get() + k * pkernel_stride,
                                   kernels   + k * hkernel_stride,
                                   pkernel_stride * sizeof(float),
                                   cudaMemcpyHostToDevice) );
                }
            }

            partial_->forward(in_d.get(), out_d.get(),
                              kernels_d.get(), 1, workspace_d);
        }

        full_->apply_bias(out_d.get(), biases_d.get());

        checkCudaErrors( cudaMemcpy(out, out_d.get(),
                                    output_len * sizeof(float),
                                    cudaMemcpyDeviceToHost) );
    }

public:
    in_split_cudnn_convolutional_layer( handle_t& handle,
                                        long_t fin,
                                        long_t fin_chunk,
                                        long_t fout,
                                        vec3i const & is,
                                        vec3i const & ks )
        : gpuram_layer_base(1, fin, fout, is, ks)
    {
        n_full_ = fin / fin_chunk;

        full_ = std::unique_ptr<native_cudnn_convolutional_layer>
            ( new native_cudnn_convolutional_layer
              (handle, fin_chunk, fout, is, ks));

        workspace_size_ = full_->workspace_size();

        if ( fin % fin_chunk )
        {
            partial_ = std::unique_ptr<native_cudnn_convolutional_layer>
                ( new native_cudnn_convolutional_layer
                  (handle, fin%fin_chunk, fout, is, ks));

            workspace_size_ = std::max(workspace_size_,
                                       partial_->workspace_size());
        }
    }

};

}}} // namespace znn::fwd::gpu
