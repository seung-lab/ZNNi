#pragma once

#include "znn/fusion/ram/native/fft_conv.hpp"
#include "znn/fusion/ram/native/cudnn_conv.hpp"

#include <vector>

namespace znn { namespace fwd { namespace device { namespace fusion {
namespace ram {

class in_split_conv: public conv_layer<layer>
{
private:
    conv_layer<layer> full_desc;
    conv_layer<layer> part_desc;

    std::vector<std::unique_ptr<device_tensor<float,5>>> full_kernels;
    std::vector<std::unique_ptr<device_tensor<float,5>>> part_kernels;

    device_array<float> all_biases;

    size_t workspace_size_ = 0;

    std::unique_ptr<native::native_conv_layer> full_;
    std::unique_ptr<native::native_conv_layer> part_;

private:
    void get_full()
    {
        // try cudnn
        full_ = make_unique<native::cudnn_conv>
            (1, full_desc.num_inputs, full_desc.num_outputs,
             full_desc.in_image_size, full_desc.kernel_size);

        std::cout << "herexx" << std::endl;

        double t1 = full_->estimate_runtime(full_kernels.size());

        std::cout << "here " << t1 << std::endl;

        auto full = make_unique<native::fft_conv>
            (1, full_desc.num_inputs, full_desc.num_outputs,
             full_desc.in_image_size, full_desc.kernel_size);

        double t2 = full->estimate_runtime(full_kernels.size());

        if ( t2 < t1 )
        {
            full_ = std::move(full);
        }

        workspace_size_ = full_->workspace_size();
    }

    void get_part()
    {
        if ( part_kernels.size() == 0 )
        {
            return;
        }

        // try cudnn
        part_ = make_unique<native::cudnn_conv>
            (1, part_desc.num_inputs, part_desc.num_outputs,
             part_desc.in_image_size, part_desc.kernel_size);

        double t1 = part_->estimate_runtime(1);

        auto part = make_unique<native::fft_conv>
            (1, part_desc.num_inputs, part_desc.num_outputs,
             part_desc.in_image_size, part_desc.kernel_size);

        double t2 = part->estimate_runtime(full_kernels.size());

        if ( t2 < t1 )
        {
            part_ = std::move(part);
        }

        if ( part_->workspace_size() > workspace_size_ )
        {
            workspace_size_ = part_->workspace_size();
        }
    }

private:
    void init_kernels( float const * kernels )
    {
        long_t hkernel_stride = kernel_len * num_inputs;
        long_t dkernel_stride = kernel_len * full_desc.num_inputs;

        for ( size_t i = 0; i < full_kernels.size(); ++i )
        {
            full_kernels[i] = make_unique<device_tensor<float,5>>
                (full_desc.kernels_shape);

            for ( long_t o = 0; o < num_outputs; ++o )
            {
                (*(full_kernels[i]))[o].load(kernels + o * hkernel_stride,
                                             from_host);
            }

            kernels += dkernel_stride;
        }

        if ( part_kernels.size() )
        {
            part_kernels[0] = make_unique<device_tensor<float,5>>
                (part_desc.kernels_shape);

            for ( long_t o = 0; o < num_outputs; ++o )
            {
                (*(part_kernels[0]))[o].load(kernels + o * hkernel_stride,
                                             from_host);
            }
        }
    }

public:
    size_t workspace_size() const
    {
        return workspace_size_;
    }

    void forward( float* in,
                  float* out,
                  void* workspace_d ) const
    {
        device_array<float> out_d(full_desc.output_len);

        for ( size_t i = 0; i < full_kernels.size(); ++i )
        {
            device_tensor<float,5> in_d (full_desc.input_shape );
            in_d.load(in, from_host);

            full_->forward( std::move(in_d), out_d.data(),
                            (*(full_kernels[i])), (i==0)?0:1, workspace_d);

            in += full_desc.input_len;
        }

        if ( part_kernels.size() )
        {
            device_tensor<float,5> in_d ( part_desc.input_shape );
            in_d.load(in, from_host);

            part_->forward( std::move(in_d), out_d.data(),
                            (*(part_kernels[0])), 1, workspace_d );
        }

        full_->nonlineiaruty(out_d.data(), all_biases.data());

        out_d.store_n(out, output_len, to_host);
    }

public:
    in_split_conv( long_t fin,
                   long_t fin_chunk,
                   long_t fout,
                   vec3i const & is,
                   vec3i const & ks,
                   host_tensor<float,5> const & kernels,
                   host_array<float> const & biases )
        : conv_layer<layer>(1,fin,fout,is,ks)
        , full_desc(1,fin_chunk,fout,is,ks)
        , part_desc(1,fin%fin_chunk,fout,is,ks)
        , full_kernels(fin/fin_chunk)
        , part_kernels(fin%fin_chunk)
        , all_biases(fout)
    {
        all_biases = biases;
        init_kernels(kernels.data());
        get_full();
        get_part();
    }
};

}}}}} // namespace znn::fwd::device::fusion::ram
