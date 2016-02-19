#pragma once

#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../memory.hpp"
#include "../../layer.hpp"
#include "../host_layer.hpp"
#include "../handle.hpp"
#include "../utils/task_package.hpp"
#include "padded_pruned_fft/fft.hpp"

namespace znn { namespace fwd { namespace cpu {


class padded_pruned_fft_convolutional_layer
    : public cpu_convolutional_layer_base
    , public host_layer
{
private:
    padded_pruned_fft_transformer* fft_;

public:

    padded_pruned_fft_convolutional_layer( long_t n, long_t fin, long_t fout,
                                           vec3i const & is, vec3i const & ks,
                                           real * km = nullptr,
                                           real* bs = nullptr )
        : cpu_convolutional_layer_base( n, fin, fout, is, ks, km, bs )
        , fft_(padded_pruned_fft_plans.get(is,ks))
    { }


private:
    void do_input_fft( real* in,
                       complex* out,
                       void* ) const noexcept
    {
        fft_->forward_image(in,out);
    }

    void do_input_fft_padded( real* in,
                              complex* out,
                              void* scratch) const noexcept
    {
        // copy the image
        real* tmp = reinterpret_cast<real*>(scratch);

        std::memcpy( tmp, in, fft_->image_memory());

        // append zeros
        long_t zero_bytes = fft_->image_scratch_memory()
            - fft_->image_memory();
        std::memset( tmp + fft_->image_elements(), 0, zero_bytes );

        fft_->forward_image(tmp,out);
    }

    void do_output_ifft( complex* in, real* out,
                         real bias, void* stack ) const
    {
        real* out_scratch = reinterpret_cast<real*>(stack);

        fft_->backward(in,out_scratch);
        real scale = fft_->get_scale();

        long_t off = fft_->result_offset();

        for ( long_t i = 0; i < out_image_len; ++i )
        {
            // out[i] = std::max(static_cast<real>(0),
            //                   out_scratch[i+off] / scale + bias);
            out[i] = out_scratch[i+off] / scale + bias;
        }
    }

    void collect_single_kernel( bool first,
                                real * kernel,
                                real * kernel_scratch,
                                complex * input,
                                complex * output,
                                complex * output_scratch,
                                long_t input_stride,
                                long_t output_stride ) const
    {
        // copy the kernel to the scratch
        std::memcpy( kernel_scratch, kernel, fft_->kernel_memory());

        // append zeros
        long_t zero_bytes = fft_->kernel_scratch_memory()
            - fft_->kernel_memory();

        std::memset( kernel_scratch + fft_->kernel_elements(),
                     0, zero_bytes );

        // transform the kernel
        fft_->forward_kernel( kernel_scratch, output_scratch );

        // loop over the batch
        long_t n_elements = fft_->transform_elements();

        for ( long_t k = 0; k < batch_size; ++k )
        {
            complex * a = input  + k * input_stride ;
            complex * r = output + k * output_stride;
            if ( first )
            {
                for ( long_t i = 0; i < n_elements; ++i )
                    r[i] = a[i] * output_scratch[i];
            }
            else
            {
                for ( long_t i = 0; i < n_elements; ++i )
                    r[i] += a[i] * output_scratch[i];
            }
        }
    }

    void collect_single_output( long_t out_num,
                                complex* inputs,
                                complex* outputs,
                                void*) const
    {
        long_t cstride = fft_->transform_elements();

        auto output_scratch = get_array<complex>(fft_->transform_elements());
        auto kernel_scratch = get_array<real>(fft_->kernel_scratch_elements());

        real* first_kernel = kernels.get() + out_num * kernel_len * num_inputs;

        for ( long_t i = 0; i < num_inputs; ++i )
        {
            collect_single_kernel( i == 0,
                                   first_kernel + i * kernel_len,
                                   kernel_scratch.get(),
                                   inputs + i * cstride,
                                   outputs + out_num * cstride,
                                   output_scratch.get(),
                                   num_inputs * cstride,
                                   num_outputs * cstride );
        }
    }


private:
    host_array<complex> transform_inputs( host_array<real> in ) const
    {
        long_t relements = fft_->image_elements();
        long_t celements = fft_->transform_elements();

        auto itransforms
            = get_array<complex>(batch_size * num_inputs * celements);

        auto fn = fft_->needs_padding() ?
            &padded_pruned_fft_convolutional_layer::do_input_fft_padded :
            &padded_pruned_fft_convolutional_layer::do_input_fft;

        for ( long_t i = 0, off = 0; i < batch_size; ++i )
            for ( long_t j = 0; j < num_inputs; ++j, ++off )
            {
                handle.add_task( fn, this,
                                 in.get() + relements * off,
                                 itransforms.get() + celements * off );
            }

        handle.execute( fft_->needs_padding() ?
                        fft_->image_scratch_memory() : 0);

        return itransforms;
    }

    host_array<complex> collect_outputs( host_array<complex> itransforms ) const
    {
        long_t celements = fft_->transform_elements();

        auto otransforms
            = get_array<complex>(batch_size * num_outputs * celements);

        auto fn = &padded_pruned_fft_convolutional_layer::collect_single_output;

        for ( long_t i = 0; i < num_outputs; ++i )
        {
            handle.add_task(fn, this, i, itransforms.get(), otransforms.get());
        }

        handle.execute();
        return otransforms;
    }

    host_array<real> process_outputs( host_array<complex> otransforms ) const
    {
        long_t celements = fft_->transform_elements();

        auto result = get_array<real>(total_output_len);

        auto fn = &padded_pruned_fft_convolutional_layer::do_output_ifft;

        for ( long_t i = 0, off = 0; i < batch_size; ++i )
        {
            for ( long_t j = 0; j < num_outputs; ++j, ++off )
            {
                handle.add_task( fn, this,
                                 otransforms.get() + celements * off,
                                 result.get() + out_image_len * off,
                                 biases.get()[j] );
            }
        }

        handle.execute( fft_->result_scratch_memory() );

        return result;
    }

public:
    host_array<real> forward( host_array<real> in ) const override
    {
        auto   in_transforms  = transform_inputs(std::move(in));
        auto   out_transforms = collect_outputs(std::move(in_transforms));
        return process_outputs(std::move(out_transforms));
    }


};



}}} // namespace znn::fwd::cpu
