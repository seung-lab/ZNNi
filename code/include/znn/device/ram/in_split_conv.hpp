#pragma once

#include "znn/types.hpp"
#include "znn/layer.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/common/utils.hpp"
#include "znn/device/common/cudnn.hpp"
#include "znn/device/common/handle.hpp"

namespace znn { namespace fwd { namespace device { namespace ram {

template<typename Conv>
class in_split_conv
    : public conv_layer<layer>
{
private:
    std::unique_ptr<Conv> full_   ;
    std::unique_ptr<Conv> partial_;

    long_t n_full_;
    long_t workspace_size_ = 0;

public:
    long_t workspace_size() const
    {
        return workspace_size_;
    }

    void forward( float const * in,
                  float* out,
                  float const * kernels,
                  float const * biases,
                  void * workspace_d) const
    {
        device_array<float> out_d    (full_->output_len);
        device_array<float> kernels_d(full_->kernels_len);
        device_array<float> biases_d (full_->num_outputs);

        biases_d.load(biases, from_host);

        long_t hkernel_stride = kernel_len * num_inputs;
        long_t dkernel_stride = kernel_len * full_->num_inputs;


        for ( long_t i = 0; i < n_full_; ++i )
        {
            device_tensor<float,5> in_d(full_->input_shape);
            in_d.load(in, from_host);

            // copy the kernels
            if ( hkernel_stride == dkernel_stride )
            {
                kernels_d.load(kernels, from_host);
            }
            else
            {
                for ( long_t k = 0; k < num_outputs; ++k )
                {
                    checkCudaErrors(
                        cudaMemcpy(kernels_d.data() + k * dkernel_stride,
                                   kernels          + k * hkernel_stride,
                                   dkernel_stride * sizeof(float),
                                   cudaMemcpyHostToDevice) );
                }
            }

            full_->forward( std::move(in_d), out_d.data(),
                            kernels_d.data(), (i==0)?0:1, workspace_d);

            in      += full_->input_len;
            kernels += dkernel_stride;
        }

        if ( partial_ )
        {
            long_t pkernel_stride = kernel_len * partial_->num_inputs;

            device_tensor<float,5> in_d(partial_->input_shape);
            in_d.load(in, from_host);

            // copy the kernels
            {
                for ( long_t k = 0; k < num_outputs; ++k )
                {
                    checkCudaErrors(
                        cudaMemcpy(kernels_d.data() + k * pkernel_stride,
                                   kernels          + k * hkernel_stride,
                                   pkernel_stride * sizeof(float),
                                   cudaMemcpyHostToDevice) );
                }
            }

            partial_->forward( std::move(in_d), out_d.data(),
                               kernels_d.data(), 1, workspace_d );
        }

        full_->nonlineiaruty(out_d.data(), biases_d.data());

        checkCudaErrors( cudaMemcpy(out, out_d.data(),
                                    output_len * sizeof(float),
                                    cudaMemcpyDeviceToHost) );
    }

public:
    in_split_conv( long_t fin, long_t fin_chunk, long_t fout,
                   vec3i const & is, vec3i const & ks )
        : conv_layer<layer>(1, fin, fout, is, ks)
    {
        n_full_ = fin / fin_chunk;

        full_ = make_unique<Conv>(1, fin_chunk, fout, is, ks);

        workspace_size_ = full_->workspace_size();

        if ( fin % fin_chunk )
        {
            partial_ = make_unique<Conv>(1, fin%fin_chunk, fout, is, ks);
            workspace_size_ = std::max(workspace_size_,
                                       partial_->workspace_size());
        }
    }

};

}}}} // namespace znn::fwd::device::ram
