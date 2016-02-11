#pragma once

#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../memory.hpp"
#include "../../layer.hpp"
#include "../host_layer.hpp"
#include "../utils/task_package.hpp"
#include "padded_pruned_fft/fft.hpp"

namespace znn { namespace fwd { namespace cpu {


class padded_pruned_parallel_fft_convolutional_layer
    : public cpu_convolutional_layer_base
    , public host_layer
{
private:
    task_package &                 handle_;
    padded_pruned_fft_transformer* fft_;

public:

    padded_pruned_parallel_fft_convolutional_layer
    ( task_package& handle,
      long_t n, long_t fin, long_t fout,
      vec3i const & is, vec3i const & ks,
      real * km = nullptr,
      real* bs = nullptr )
        : cpu_convolutional_layer_base( n, fin, fout, is, ks, km, bs )
        , handle_(handle)
        , fft_(padded_pruned_fft_plans.get(is,ks))
    {
        STRONG_ASSERT(n==1);
    }


private:
    void parallel_bias( real* in, real* out,
                        real scale, real bias,
                        long_t len ) const
    {
        long_t chunk_len = 3 * len / handle_.concurrency() / 2;

        chunk_len = std::max(chunk_len, static_cast<long_t>(1));
        chunk_len = std::min(chunk_len, len);

        long_t n_chunks = len / chunk_len;
        long_t left_len = len % chunk_len;

        for ( long_t i = 0; i < n_chunks; ++i )
        {
            long_t first = i * chunk_len;
            long_t last  = first + chunk_len;

            handle_.add_task([in,out,scale,bias,first,last](void*){
                    for ( long_t j = first; j < last; ++j )
                        out[j] = in[j] / scale + bias;
                });
        }

        if ( left_len )
        {
            long_t first = n_chunks * chunk_len;
            long_t last  = first + left_len;

            handle_.add_task([in,out,scale,bias,first,last](void*){
                    for ( long_t j = first; j < last; ++j )
                        out[j] = in[j] / scale + bias;
                });
        }

        handle_.execute();
    }

    void parallel_mad( complex* in1, complex* in2,
                       complex* out, long_t len ) const
    {
        long_t chunk_len = 3 * len / handle_.concurrency() / 2;

        chunk_len = std::max(chunk_len, static_cast<long_t>(1));
        chunk_len = std::min(chunk_len, len);

        long_t n_chunks = len / chunk_len;
        long_t left_len = len % chunk_len;

        for ( long_t i = 0; i < n_chunks; ++i )
        {
            long_t first = i * chunk_len;
            long_t last  = first + chunk_len;

            handle_.add_task([in1,in2,out,first,last](void*){
                    for ( long_t j = first; j < last; ++j )
                        out[j] += in1[j] * in2[j];
                });
        }

        if ( left_len )
        {
            long_t first = n_chunks * chunk_len;
            long_t last  = first + left_len;

            handle_.add_task([in1,in2,out,first,last](void*){
                    for ( long_t j = first; j < last; ++j )
                        out[j] += in1[j] * in2[j];
                });
        }

        handle_.execute();
    }

    void parallel_mset( complex* in1, complex* in2,
                        complex* out, long_t len ) const
    {
        long_t chunk_len = 3 * len / handle_.concurrency() / 2;

        chunk_len = std::max(chunk_len, static_cast<long_t>(1));
        chunk_len = std::min(chunk_len, len);

        long_t n_chunks = len / chunk_len;
        long_t left_len = len % chunk_len;

        for ( long_t i = 0; i < n_chunks; ++i )
        {
            long_t first = i * chunk_len;
            long_t last  = first + chunk_len;

            handle_.add_task([in1,in2,out,first,last](void*){
                    for ( long_t j = first; j < last; ++j )
                        out[j] = in1[j] * in2[j];
                });
        }

        if ( left_len )
        {
            long_t first = n_chunks * chunk_len;
            long_t last  = first + left_len;

            handle_.add_task([in1,in2,out,first,last](void*){
                    for ( long_t j = first; j < last; ++j )
                        out[j] = in1[j] * in2[j];
                });
        }

        handle_.execute();
    }

    void do_input_fft( real* in, complex* out ) const noexcept
    {
        fft_->parallel_forward_image(handle_, in, out);
    }

    void do_input_fft_padded( real* in,
                              complex* out,
                              real* scratch) const noexcept
    {
        // copy the image
        std::memcpy( scratch, in, fft_->image_memory());

        // append zeros
        long_t zero_bytes = fft_->image_scratch_memory()
            - fft_->image_memory();

        std::memset( scratch + fft_->image_elements(), 0, zero_bytes );

        fft_->parallel_forward_image(handle_, scratch, out);
    }

    void do_output_ifft( complex* in,
                         real*    out,
                         real     bias,
                         real*    rscratch ) const
    {
        fft_->parallel_backward( handle_, in, rscratch );

        real scale = fft_->get_scale();
        long_t off = fft_->result_offset();

        // TODO: PARALLEL
        // for ( long_t i = 0; i < out_image_len; ++i )
        // {
        //     // out[i] = std::max(static_cast<real>(0),
        //     //                   out_scratch[i+off] / scale + bias);
        //     out[i] = rscratch[i+off] / scale + bias;
        // }

        parallel_bias( rscratch + off, out, scale, bias, out_image_len );
    }

    void collect_single_kernel( bool first,
                                real * kernel,
                                real * rscratch,
                                complex * input,
                                complex * output,
                                complex * cscratch ) const
    {
        // copy the kernel to the scratch
        std::memcpy( rscratch, kernel, fft_->kernel_memory());

        // append zeros
        long_t zero_bytes = fft_->kernel_scratch_memory()
            - fft_->kernel_memory();

        std::memset( rscratch + fft_->kernel_elements(), 0, zero_bytes );

        // transform the kernel
        fft_->parallel_forward_kernel( handle_, rscratch, cscratch );

        // loop over the batch
        long_t n_elements = fft_->transform_elements();

        // TODO: PARALLEL
        if ( first )
        {
            parallel_mset(input, cscratch, output, n_elements);
        }
        else
        {
            parallel_mad(input, cscratch, output, n_elements);
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



}}} // namespace znn::fwd::cpu
