#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/host_layer.hpp"
#include "znn/host/v1/conv_data.hpp"
#include "znn/host/common/fft/fft.hpp"
#include "znn/host/common/thread_pin.hpp"

#include <atomic>
#include <thread>
#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace host { namespace v1 {

class fft_conv
    : public conv_layer<host_layer>
    , public conv_data
{
private:
    padded_pruned_fft_transformer* fft_;

public:
    fft_conv( long_t n, long_t fin, long_t fout,
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
                                complex* outputs,
                                int core_id ) const noexcept
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

        // figure out an optimal way to saturate the cores

        long_t n_cores = std::thread::hardware_concurrency() / 2;

        // simple strategy.. all batches on the same core
        if ( (n_cores <= num_outputs) || (in_batch_size == 1) )
        {
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
        else
        {
            long_t tasks_per_core   = (n_cores + num_outputs - 1) / num_outputs;
            long_t batches_per_core = in_batch_size / tasks_per_core;

            batches_per_core
                = std::max(batches_per_core, static_cast<long_t>(1));

            tasks_per_core
                = ( in_batch_size + batches_per_core - 1 ) / batches_per_core;

            tbb::task_group tg;

            for ( long_t t = 0; t < tasks_per_core; ++t )
            {
                auto fn = [=]()
                {
                    cpu_pin pin(core_id);
                    for ( long_t k = t * batches_per_core;
                          k < std::min( (t+1) * batches_per_core, in_batch_size );
                          ++k )
                    {
                        complex* a = input  + k * input_stride ;
                        complex* r = output + k * output_stride;

                        if ( in_no == 0 )
                            this->mul_to(a, oscratch, r, n_elements);
                        else
                            this->mul_add_to(a, oscratch, r, n_elements);
                    }
                };

                if ( t < tasks_per_core - 1 )
                {
                    tg.run(fn);
                }
                else
                {
                    tg.run_and_wait(fn);
                }
            }
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

        // create the list of all transforms to be processed
        std::vector<std::pair<real*, complex*>>
            all_transforms(num_inputs*in_batch_size);

        tbb::concurrent_queue<std::pair<real*, complex*>*> queue;

        for ( long_t i = 0, off = 0; i < in_batch_size; ++i )
            for ( long_t j = 0; j < num_inputs; ++j, ++off )
            {
                all_transforms[off].first  = in.data() + relements*off;
                all_transforms[off].second = itransforms.data() + celements*off;
                queue.push(&all_transforms[off]);
            }

        auto fn = [&,this]()
            {
                host_tensor<real,1> scratch;

                if ( this->fft_->needs_padding() )
                {
                    scratch = host_tensor<real,1>
                        (this->fft_->image_scratch_elements());
                }

                std::pair<real*, complex*>* which;

                if ( !this->fft_->needs_padding() )
                {
                    while ( queue.try_pop(which) )
                    {
                        this->do_input_fft(which->first,
                                           which->second);
                    }
                }
                else
                {
                    while ( queue.try_pop(which) )
                    {
                        this->do_input_fft_padded(which->first,
                                                  which->second,
                                                  scratch.data());
                    }
                }
            };


        tbb::task_group tg;

        long_t num_tasks = std::thread::hardware_concurrency();
        num_tasks = std::min(num_tasks, in_batch_size*num_inputs);

        for ( long_t i = 0; i < num_tasks - 1; ++i )
        {
            tg.run(fn);
        }

        tg.run_and_wait(fn);

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



        std::vector<tbb::concurrent_queue<long_t>> queues(num_inputs);

        for ( long_t i = 0; i < num_outputs; ++i )
        {
            queues[0].push(i);
        }

        thread_distributor td;

        auto fn = [&,this]()
            {
                thread_pin pin(td);

                host_tensor<complex,1> oscratch(this->fft_->transform_elements());
                host_tensor<real,1>    iscratch(this->fft_->kernel_scratch_elements());

                long_t out_no;

                for ( long_t i = 0; i < num_inputs; ++i )
                {
                    while ( queues[i].try_pop(out_no) )
                    {
                        this->process_single_kernel( i, out_no,
                                                     iscratch.data(),
                                                     oscratch.data(),
                                                     itransforms.data(),
                                                     otransforms.data(),
                                                     pin.location());
                        if ( i < num_inputs - 1 )
                        {
                            queues[i+1].push(out_no);
                        }
                    }
                }
            };


        long_t num_tasks = std::thread::hardware_concurrency();
        num_tasks = std::min(num_tasks, num_outputs);

        tbb::task_group tg;

        for ( long_t i = 0; i < num_tasks - 1; ++i )
        {
            tg.run(fn);
        }

        tg.run_and_wait(fn);

        return otransforms;
    }

    host_tensor<real,5>
    process_outputs( host_tensor<complex,5> otransforms ) const
    {
        long_t celements = fft_->transform_elements();

        host_tensor<real,5> result
            (out_batch_size, num_outputs,
             out_image_size[0], out_image_size[1], out_image_size[2]);

        // create the list of all transforms to be processed
        std::vector<std::tuple<real, complex*, real*>>
            all_transforms(num_outputs*in_batch_size);

        tbb::concurrent_queue<std::tuple<real, complex*, real*>*> queue;

        for ( long_t i = 0, off = 0; i < in_batch_size; ++i )
            for ( long_t j = 0; j < num_outputs; ++j, ++off )
            {
                std::get<0>(all_transforms[off]) = biases.data()[j];
                std::get<1>(all_transforms[off])
                    = otransforms.data() + celements * off;
                std::get<2>(all_transforms[off])
                    = result.data() + out_image_len * off;
                queue.push(&all_transforms[off]);
            }


        auto fn = [&,this]()
        {
            host_tensor<real,1> scratch(this->fft_->result_scratch_elements());

            std::tuple<real, complex*, real*>* which;

            while ( queue.try_pop(which) )
            {
                this->do_output_ifft( std::get<1>(*which),
                                      std::get<2>(*which),
                                      std::get<0>(*which),
                                      scratch.data() );
            }
        };

        tbb::task_group tg;

        long_t num_tasks = std::thread::hardware_concurrency();
        num_tasks = std::min(num_tasks, in_batch_size*num_inputs);

        for ( long_t i = 0; i < num_tasks - 1; ++i )
        {
            tg.run(fn);
        }

        tg.run_and_wait(fn);

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
        long_t num_tasks = std::thread::hardware_concurrency();
        num_tasks = std::min(num_tasks, in_batch_size*num_inputs);

        vec3i ts = fft_->transform_size();

        long_t stage1_memory = input_memory +
            in_batch_size * num_inputs * ts[0]*ts[1]*ts[2] * sizeof(complex);

        if ( fft_->needs_padding() )
        {
            stage1_memory
                += num_tasks * fft_->image_scratch_elements() * sizeof(real);
        }

        num_tasks = std::thread::hardware_concurrency();
        num_tasks = std::min(num_tasks, num_outputs);

        long_t stage2_memory
            = in_batch_size * num_inputs * ts[0]*ts[1]*ts[2] * sizeof(complex)
            + out_batch_size * num_outputs * ts[0]*ts[1]*ts[2] * sizeof(complex)
            + num_tasks * fft_->transform_elements() * sizeof(complex)
            + num_tasks * fft_->kernel_scratch_elements() * sizeof(real);

        num_tasks = std::thread::hardware_concurrency();
        num_tasks = std::min(num_tasks, in_batch_size*num_inputs);

        long_t stage3_memory
            = out_batch_size * num_outputs * ts[0]*ts[1]*ts[2] * sizeof(complex)
            + output_memory
            + fft_->result_scratch_elements() * num_tasks * sizeof(real);

        return std::max({stage1_memory,stage2_memory,stage3_memory});
    }

};


}}}} // namespace znn::fwd::host::v1
