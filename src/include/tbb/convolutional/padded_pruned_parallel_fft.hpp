#pragma once

#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../memory.hpp"
#include "../../layer.hpp"
#include "../host_layer.hpp"
#include "padded_pruned_fft/fft.hpp"
#include "base.hpp"
#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace tbb {


class padded_pruned_parallel_fft_convolutional_layer
    : public cpu_convolutional_layer_base
    , public host_layer
{
private:
    padded_pruned_fft_transformer* fft_;

public:

    padded_pruned_parallel_fft_convolutional_layer
    ( void*,
      long_t n, long_t fin, long_t fout,
      vec3i const & is, vec3i const & ks,
      real * km = nullptr,
      real* bs = nullptr )
        : cpu_convolutional_layer_base( n, fin, fout, is, ks, km, bs )
        , fft_(padded_pruned_fft_plans.get(is,ks))
    {
        STRONG_ASSERT(n==1);
    }


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
        ::tbb::parallel_for( static_cast<long_t>(0), fft_->image_elements(),
                             [&](long_t i)
                             {
                                 scratch[i] = in[i];
                             } );

        // append zeros
        ::tbb::parallel_for( fft_->image_elements(),
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

        ::tbb::parallel_for( static_cast<long_t>(0), out_image_len,
                             [&](long_t i)
                             {
                                 out[i] = rin[i] / scale + bias;
                             });
    }

    void collect_single_kernel( bool first,
                                real * kernel,
                                real * rscratch,
                                complex * input,
                                complex * output,
                                complex * cscratch ) const
    {
        // copy the kernel to the scratch
        ::tbb::parallel_for( static_cast<long_t>(0), fft_->kernel_elements(),
                             [&](long_t i)
                             {
                                 rscratch[i] = kernel[i];
                             } );

        // append zeros
        ::tbb::parallel_for( fft_->kernel_elements(),
                             fft_->kernel_scratch_elements(),
                             [&](long_t i)
                             {
                                 rscratch[i] = 0;
                             } );

        // transform the kernel
        fft_->parallel_forward_kernel( rscratch, cscratch );

        // loop over the batch
        long_t n_elements = fft_->transform_elements();

        // TODO: PARALLEL
        if ( first )
        {
            ::tbb::parallel_for( static_cast<long_t>(0), n_elements,
                                 [&](long_t i)
                                 {
                                     output[i] = input[i] * cscratch[i];
                                 } );
        }
        else
        {
            ::tbb::parallel_for( static_cast<long_t>(0), n_elements,
                                 [&](long_t i)
                                 {
                                     output[i] += input[i] * cscratch[i];
                                 } );
        }
    }

    void collect_single_output( real*    kernelz,
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
    host_array<complex> transform_inputs( host_array<real> in ) const
    {
        long_t relements = fft_->image_elements();
        long_t celements = fft_->transform_elements();

        auto itransforms
            = get_array<complex>(batch_size * num_inputs * celements);

        if ( fft_->needs_padding() )
        {
            auto scratch = get_array<real>(fft_->image_scratch_elements());
            for ( long_t j = 0; j < num_inputs; ++j )
            {
                do_input_fft_padded( in.get() + relements * j,
                                     itransforms.get() + celements * j,
                                     scratch.get() );
            }
        }
        else
        {
            for ( long_t j = 0; j < num_inputs; ++j )
            {
                do_input_fft( in.get() + relements * j,
                              itransforms.get() + celements * j);
            }
        }
        return itransforms;
    }

    host_array<real> collect_outputs( host_array<complex> itransforms ) const
    {
        long_t celements = fft_->transform_elements();
        long_t relements = std::max(fft_->result_scratch_elements(),
                                    fft_->kernel_scratch_elements());

        auto result = get_array<real>(total_output_len);

        auto c1scratch = get_array<complex>(celements);
        auto c2scratch = get_array<complex>(celements);

        auto rscratch  = get_array<real>(relements);

        for ( long_t i = 0; i < num_outputs; ++i )
        {
            collect_single_output( kernels.get() + kernel_len * num_inputs * i,
                                   itransforms.get(),
                                   c2scratch.get(),
                                   rscratch.get(),
                                   c1scratch.get() );

            do_output_ifft( c2scratch.get(),
                            result.get() + out_image_len * i,
                            biases.get()[i],
                            rscratch.get() );
        }

        return result;
    }


public:
    host_array<real> forward( host_array<real> in ) const override
    {
        auto   in_transforms  = transform_inputs(std::move(in));
        return collect_outputs(std::move(in_transforms));
    }
};



}}} // namespace znn::fwd::tbb
