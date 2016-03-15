#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/host_layer.hpp"
#include "znn/host/v1/conv_data.hpp"
#include "znn/host/common/fft/fft.hpp"

#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace host { namespace v1 {

class dp_fft_conv
    : public conv_layer<host_layer>
    , public conv_data
{
private:
    padded_pruned_fft_transformer* fft_;

public:
    dp_fft_conv( long_t n, long_t fin, long_t fout,
                 vec3i const & is, vec3i const & ks,
                 float * km = nullptr, float* bs = nullptr )
        : conv_layer<host_layer>(n,fin,fout,is,ks)
        , conv_data(fin,fout,ks,km,bs)
        , fft_(padded_pruned_fft_plans.get(is,ks))
    { }


private:
    void do_input_fft( real* in, complex* out ) const noexcept
    {
        fft_->parallel_forward_image(in, out);
    }

    void do_input_fft_padded( real* in,
                              complex* out,
                              real* scratch) const noexcept
    {
        // copy the image to the scratch
        tbb::parallel_for( static_cast<long_t>(0), fft_->image_elements(),
                           [&](long_t i)
                           {
                               scratch[i] = in[i];
                           } );

        // append zeros
        tbb::parallel_for( fft_->image_elements(),
                           fft_->image_scratch_elements(),
                           [&](long_t i)
                           {
                               scratch[i] = 0;
                           } );


        fft_->parallel_forward_image(scratch, out);
    }

    void do_output_ifft( complex* in,
                         real*    out,
                         real     bias,
                         real*    rscratch ) const
    {
        fft_->parallel_backward( in, rscratch );

        real scale = fft_->get_scale();
        long_t off = fft_->result_offset();

        real* rin = rscratch + off;

        tbb::parallel_for( static_cast<long_t>(0), out_image_len,
                           [&](long_t i)
                           {
                               out[i] = rin[i] / scale + bias;
                           });
    }

    void collect_single_kernel( bool first,
                                real const * kernel,
                                real * rscratch,
                                complex * inputs,
                                complex * outputs,
                                complex * cscratch ) const
    {
        // copy the kernel to the scratch
        tbb::parallel_for( static_cast<long_t>(0), fft_->kernel_elements(),
                           [&](long_t i)
                           {
                               rscratch[i] = kernel[i];
                           } );

        // append zeros
        tbb::parallel_for( fft_->kernel_elements(),
                           fft_->kernel_scratch_elements(),
                           [&](long_t i)
                           {
                               rscratch[i] = 0;
                           } );

        // transform the kernel
        fft_->parallel_forward_kernel( rscratch, cscratch );

        // loop over the batch
        long_t n_elements = fft_->transform_elements();

        complex * input  = inputs ;
        complex * output = outputs;

        for ( long_t i = 0; i < in_batch_size; ++i )
        {
            if ( first )
            {
                tbb::parallel_for( static_cast<long_t>(0), n_elements,
                                   [&](long_t i)
                                   {
                                       output[i] = input[i] * cscratch[i];
                                   } );
            }
            else
            {
                tbb::parallel_for( static_cast<long_t>(0), n_elements,
                                   [&](long_t i)
                                   {
                                       output[i] += input[i] * cscratch[i];
                                   } );
            }

            input  += n_elements * num_inputs;
            output += n_elements;
        }
    }

    void collect_single_output( real const * kernelz,
                                complex* inputs,
                                complex* output,
                                real *   rscratch,
                                complex* cscratch ) const
    {
        long_t cstride = fft_->transform_elements();

        for ( long_t i = 0; i < num_inputs; ++i )
        {
            collect_single_kernel( i == 0,
                                   kernelz + i * kernel_len,
                                   rscratch,
                                   inputs + i * cstride,
                                   output,
                                   cscratch );
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

        if ( fft_->needs_padding() )
        {
            host_array<real> scratch(fft_->image_scratch_elements());
            for ( long_t j = 0; j < num_inputs * in_batch_size; ++j )
            {
                do_input_fft_padded( in.data() + relements * j,
                                     itransforms.data() + celements * j,
                                     scratch.data() );
            }
        }
        else
        {
            for ( long_t j = 0; j < num_inputs * in_batch_size; ++j )
            {
                do_input_fft( in.data() + relements * j,
                              itransforms.data() + celements * j);
            }
        }
        return itransforms;
    }

    host_tensor<float,5>
    collect_outputs( host_tensor<complex,5> itransforms ) const
    {
        long_t celements = fft_->transform_elements();
        long_t relements = std::max(fft_->result_scratch_elements(),
                                    fft_->kernel_scratch_elements());

        host_tensor<real,5> result
            (out_batch_size, num_outputs,
             out_image_size[0], out_image_size[1], out_image_size[2]);

        host_array<complex> c1scratch(celements);
        host_array<complex> c2scratch(celements * in_batch_size);

        host_array<real> rscratch(relements);

        for ( long_t i = 0; i < num_outputs; ++i )
        {
            collect_single_output( kernels.data() + kernel_len * num_inputs * i,
                                   itransforms.data(),
                                   c2scratch.data(),
                                   rscratch.data(),
                                   c1scratch.data() );


            for ( long_t j = 0; j < in_batch_size; ++j )
            {
                do_output_ifft( c2scratch.data() + j * celements,
                                result.data() + out_image_len * i
                                + j * output_len,
                                biases.data()[i],
                                rscratch.data() );
            }
        }

        return result;
    }


public:
    host_tensor<real,5> forward( host_tensor<float,5> in ) const override
    {
        auto   in_transforms  = transform_inputs(std::move(in));
        return collect_outputs(std::move(in_transforms));
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

        long_t celements = fft_->transform_elements();
        long_t relements = std::max(fft_->result_scratch_elements(),
                                    fft_->kernel_scratch_elements());

        long_t stage2_memory
            = in_batch_size * num_inputs * ts[0]*ts[1]*ts[2] * sizeof(complex)
            + output_memory
            + relements * sizeof(real)
            + (in_batch_size + 1) * celements * sizeof(complex);

        return std::max(stage1_memory, stage2_memory);
    }

};


}}}} // namespace znn::fwd::host::v1
