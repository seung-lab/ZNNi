#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/2dv2/host_layer.hpp"
#include "znn/host/2dv2/conv_data.hpp"
#include "znn/host/common/fft2d/fft.hpp"
#include "znn/host/common/thread_pin.hpp"

#include <atomic>
#include <thread>
#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace host { namespace twod {

class fft_conv2d_serial
    : public conv_layer2d<host_layer2d>
    , public conv_data2d
{
private:
    padded_pruned_fft2d_transformer* fft_;
    host_tensor<complex,4> kffts_;

public:
    fft_conv2d_serial( long_t n, long_t fin, long_t fout,
                       vec2i const & is, vec2i const & ks,
                       float * km = nullptr, float* bs = nullptr )
        : conv_layer2d<host_layer2d>(n,fin,fout,is,ks)
        , conv_data2d(fin,fout,ks,km,bs)
        , fft_(padded_pruned_fft2d_plans.get(is,ks))
        , kffts_(fout,fin,fft_->transform_size()[0],fft_->transform_size()[1]);

    {
        long_t total = fin * fout;
        long_t zeros = fft_->kernel_scratch_memory() - fft_->kernel_memory();

        auto fn = [&]( long_t i )
            {
                host_array<char> scratch(fft_->kernel_scratch_memory());
                real const * kernel = kernels.data() + i*kernel_len;
                complex * ckernel = kffts_.data() + i*fft_->transform_elements();

                std::memcpy( scratch, kernel, fft_->kernel_memory());
                std::memset( scratch + fft_->kernel_elements(), 0, zeros );

                fft_->forward_kernel( scratch, ckernel );
            };

        tbb::parallel_for( static_cast<long_t>(0), total, fn );
    }


private:
    void do_input_fft( real* in, complex* out ) const noexcept
    {
        fft_->forward_image(in,out);
    }

    void do_input_fft_padded( real* in,
                              complex* out,
                              real* tmp) const noexcept
    {
        std::memcpy( tmp, in, fft_->image_memory());

        // append zeros
        long_t zero_bytes = fft_->image_scratch_memory() - fft_->image_memory();

        std::memset( tmp + fft_->image_elements(), 0, zero_bytes );

        fft_->forward_image(tmp,out);
    }

    void do_output_ifft( complex* in, real* out,
                         real bias, real* out_scratch ) const noexcept
    {
        fft_->backward(in,out_scratch);
        real scale = fft_->get_scale();

        long_t off = fft_->result_offset();

        for ( long_t i = 0; i < out_image_len; ++i )
        {
            out[i] = out_scratch[i+off] / scale + bias;
        }
    }

    void mul_to( complex* a, complex* b, complex* r, long_t n ) const noexcept
    {
        for ( long_t i = 0; i < n; ++i )
        {
            r[i] = a[i] * b[i];
        }
    }

    void mul_add_to( complex* a, complex* b,
                     complex* r, long_t n ) const noexcept
    {
        for ( long_t i = 0; i < n; ++i )
        {
            r[i] += a[i] * b[i];
        }
    }


private:
    void transform_inputs( float* in, complex* itransforms,
                           float* scratch ) const noexcept
    {
        long_t relements = fft_->image_elements();
        long_t celements = fft_->transform_elements();

        long_t total = num_inputs;

        if ( !this->fft_->needs_padding() )
        {
            for ( long_t i = 0, i < total; ++i )
                this->do_input_fft(in + relements * i,
                                   itransforms + celements * i);
        }
        else
        {
            for ( long_t i = 0, i < total; ++i )
                this->do_input_fft_padded(in + relements * i,
                                          itransforms + celements * i,
                                          scratch);
        }
    }

    void collect_outputs( complex* in, float* out,
                          complex* sum, float* scratch ) const noexcept
    {
        long_t celements = fft_->transform_elements();

        for ( long_t o = 0; o < num_outputs; ++o )
        {
            mul_to(in, kffts_[o][i].data(), sum, celements);

            for ( long_t i = 1; i < num_inputs; ++i )
            {
                mul_add_to(in + i * celements, kffts_[o][i].data(),
                           sum, celements);
            }

            do_output_ifft( sum, out + o * out_image_len,
                            biases.data()[o], scratch );
        }
    }



public:
    void forward( float* in, float* out, void* ws ) const override
    {
        complex * itransforms = reinterpret_cast<complex*>(ws);
        complex * sum = itransforms + fft_->transform_elements() * num_inputs;
        float * scratch
            = reinterpret_cast<float*>(sum + fft_->transform_elements());

        for ( long_t i = 0; i < batch_size; ++i )
        {
            transform_inputs(in, itransforms, scratch);
            collect_outputs(itransforms, out, sum, scratch);

            in  += input_len ;
            out += output_len;
        }

        auto   in_transforms  = transform_inputs(std::move(in));
        auto   out_transforms = collect_outputs(std::move(in_transforms));
        return process_outputs(std::move(out_transforms));
    }

    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory +
            num_inputs * num_outputs * fft_->transform_elements()
            * sizeof(complex);
    }

    long_t working_memory() const override
    {
        return 0;
    }

    virtual long_t workspace_size() const
    {
        return (num_inputs + 2) * fft_->transform_elements() * sizeof(complex);
    }

};


}}}} // namespace znn::fwd::host::twod
