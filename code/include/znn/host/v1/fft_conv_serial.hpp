#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/host_layer.hpp"
#include "znn/host/v1/conv_data.hpp"
#include "znn/host/common/fft/fft.hpp"

namespace znn { namespace fwd { namespace host { namespace v1 {

class fft_conv_serial
    : public conv_layer<host_layer>
    , public conv_data
{
private:
    padded_pruned_fft_transformer* fft_;

public:
    fft_conv_serial( long_t n, long_t fin, long_t fout,
                     vec3i const & is, vec3i const & ks,
                     float * km = nullptr, float* bs = nullptr )
        : conv_layer<host_layer>(n,fin,fout,is,ks)
        , conv_data(fin,fout,ks,km,bs)
        , fft_(padded_pruned_fft_plans.get(is,ks))
    { }


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

    void process_single_kernel( long_t in_no,
                                long_t out_no,
                                real* iscratch,
                                complex* oscratch,
                                complex* inputs,
                                complex* outputs ) const noexcept
    {
        real const * kernel
            = kernels.data() + (out_no*num_inputs + in_no) * kernel_len;

        // copy the kernel to the scratch
        std::memcpy( iscratch, kernel, fft_->kernel_memory());

        // append zeros
        long_t zero_bytes = fft_->kernel_scratch_memory()
            - fft_->kernel_memory();

        std::memset( iscratch + fft_->kernel_elements(), 0, zero_bytes );

        // transform the kernel
        fft_->forward_kernel( iscratch, oscratch );

        // loop over the batch
        long_t n_elements = fft_->transform_elements();

        long_t input_stride  = num_inputs  * n_elements;
        long_t output_stride = num_outputs * n_elements;

        complex* input  = inputs  + in_no  * n_elements;
        complex* output = outputs + out_no * n_elements;

        for ( long_t k = 0; k < in_batch_size; ++k )
        {
            complex* a = input  + k * input_stride ;
            complex* r = output + k * output_stride;

            if ( in_no == 0 )
                mul_to(a, oscratch, r, n_elements);
            else
                mul_add_to(a, oscratch, r, n_elements);

        }
    }

private:
    host_tensor<complex,5> transform_inputs( host_tensor<real,5> in ) const
    {
        long_t relements = fft_->image_elements();
        long_t celements = fft_->transform_elements();

        vec3i transform_size = fft_->transform_size();

        host_tensor<complex,5>
            itransforms( in_batch_size, num_inputs,
                         transform_size[0], transform_size[1],
                         transform_size[2] );

        host_tensor<real,1> scratch;

        if ( this->fft_->needs_padding() )
        {
            scratch = host_tensor<real,1>
                (this->fft_->image_scratch_elements());
        }

        for ( long_t i = 0, off = 0; i < in_batch_size; ++i )
            for ( long_t j = 0; j < num_inputs; ++j, ++off )
            {
                if ( !this->fft_->needs_padding() )
                {
                    do_input_fft(in.data() + relements*off,
                                 itransforms.data() + celements*off);
                }
                else
                {
                    do_input_fft_padded(in.data() + relements*off,
                                        itransforms.data() + celements*off,
                                        scratch.data());
                }
            }

        return itransforms;
    }

    host_tensor<complex,5>
    collect_outputs( host_tensor<complex,5> itransforms ) const
    {
        vec3i transform_size = fft_->transform_size();

        host_tensor<complex,5>
            otransforms( out_batch_size, num_outputs,
                         transform_size[0], transform_size[1],
                         transform_size[2] );

        host_tensor<complex,1> oscratch(this->fft_->transform_elements());
        host_tensor<real,1>    iscratch(this->fft_->kernel_scratch_elements());

        for ( long_t i = 0; i < num_inputs; ++i )
            for ( long_t j = 0; j < num_outputs; ++j )
                process_single_kernel( i, j,
                                       iscratch.data(),
                                       oscratch.data(),
                                       itransforms.data(),
                                       otransforms.data() );

        return otransforms;
    }

    host_tensor<real,5>
    process_outputs( host_tensor<complex,5> otransforms ) const
    {
        long_t celements = fft_->transform_elements();

        host_tensor<real,5> result
            (out_batch_size, num_outputs,
             out_image_size[0], out_image_size[1], out_image_size[2]);

        host_tensor<real,1> scratch(this->fft_->result_scratch_elements());

            // create the list of all transforms to be processed
        std::vector<std::tuple<real, complex*, real*>>
            all_transforms(num_outputs*in_batch_size);

        for ( long_t i = 0, off = 0; i < in_batch_size; ++i )
            for ( long_t j = 0; j < num_outputs; ++j, ++off )
            {
                this->do_output_ifft( otransforms.data() + celements * off,
                                      result.data() + out_image_len * off,
                                      biases.data()[j],
                                      scratch.data() );
            }
        return result;
    }

public:
    host_tensor<float,5> forward( host_tensor<float,5> in ) const override
    {
        auto   in_transforms  = transform_inputs(std::move(in));
        auto   out_transforms = collect_outputs(std::move(in_transforms));
        return process_outputs(std::move(out_transforms));
    }

    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory;
    }

    long_t working_memory() const override
    {
        vec3i ts = fft_->transform_size();

        long_t stage1_memory = input_memory +
            in_batch_size * num_inputs * ts[0]*ts[1]*ts[2] * sizeof(complex);

        if ( fft_->needs_padding() )
        {
            stage1_memory += fft_->image_scratch_elements() * sizeof(real);
        }

        long_t stage2_memory
            = in_batch_size * num_inputs * ts[0]*ts[1]*ts[2] * sizeof(complex)
            + out_batch_size * num_outputs * ts[0]*ts[1]*ts[2] * sizeof(complex)
            + fft_->transform_elements() * sizeof(complex)
            + fft_->kernel_scratch_elements() * sizeof(real);

        long_t stage3_memory
            = out_batch_size * num_outputs * ts[0]*ts[1]*ts[2] * sizeof(complex)
            + output_memory
            + fft_->result_scratch_elements() * sizeof(real);

        return std::max({stage1_memory,stage2_memory,stage3_memory});
    }

};


}}}} // namespace znn::fwd::host::v1
