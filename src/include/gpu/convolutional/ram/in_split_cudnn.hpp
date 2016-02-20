#pragma once

#include "native_cudnn.hpp"
#include "base.hpp"
#include "../../../layer.hpp"

namespace znn { namespace fwd { namespace gpu {

template<typename Native>
class in_split_cudnn_convolutional_layer
    : public convolutional_layer_base
{
private:
    std::unique_ptr<Native> full_   ;
    std::unique_ptr<Native> partial_;

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
        auto in_d      = get_device_array<float>(full_->input_len);
        auto out_d     = get_device_array<float>(full_->output_len);
        auto kernels_d = get_device_array<float>(full_->kernels_len);
        auto biases_d  = get_device_array<float>(full_->num_outputs);

        // copy the biases
        device_copy_n( biases, full_->num_outputs, biases_d );

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
    in_split_cudnn_convolutional_layer( long_t fin,
                                        long_t fin_chunk,
                                        long_t fout,
                                        vec3i const & is,
                                        vec3i const & ks )
        : convolutional_layer_base(1, fin, fout, is, ks)
    {
        n_full_ = fin / fin_chunk;

        full_ = std::unique_ptr<Native>
            ( new Native(1, fin_chunk, fout, is, ks));

        workspace_size_ = full_->workspace_size();

        if ( fin % fin_chunk )
        {
            partial_ = std::unique_ptr<Native>
                ( new Native(1, fin%fin_chunk, fout, is, ks));

            workspace_size_ = std::max(workspace_size_,
                                       partial_->workspace_size());
        }
    }

};

}}} // namespace znn::fwd::gpu
