#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/2d/host_layer.hpp"
#include "znn/host/2d/conv_data.hpp"
#include "znn/host/common/conv2d/convolver.hpp"

#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace host { namespace twod {

class direct_conv2d
    : public conv_layer2d<host_layer2d>
    , public conv_data2d
{
private:
    convolver2d convolver_;

public:
    direct_conv2d( long_t n, long_t fin, long_t fout,
                   vec2i const & is, vec2i const & ks,
                   float * km = nullptr, float* bs = nullptr )
        : conv_layer2d<host_layer2d>(n,fin,fout,is,ks)
        , conv_data2d(fin,fout,ks,km,bs)
        , convolver_(is,ks)
    { }

private:
    void nonlinearity( real* out, real bias ) const noexcept
    {
        for ( long_t i = 0; i < out_image_len; ++i )
        {
            //out[i] = std::max(static_cast<real>(0), out[i] + bias);
            out[i] = out[i] + bias;
        }
    }

#if defined(ZNN_USE_MKL_CONVOLUTION)
    void do_single_output( real* input,
                           real const * kernel,
                           real* out,
                           real  bias,
                           void* stack) const noexcept
    {
        convolver_.convolve(input, kernel, out);

        real* tmp = reinterpret_cast<real*>(stack);

        for ( long_t i = 0; i < num_inputs; ++i )
        {
            convolver_.convolve(input +  i * in_image_len,
                                    kernel + i * kernel_len,
                                    tmp);

            for ( long_t j = 0; j < out_image_len; ++j )
            {
                out[j] += tmp[j];
            }
        }

        nonlinearity(out, bias);
    }
#else
    void do_single_output( real* input ,
                           real const * kernel,
                           real* out,
                           real  bias ) const noexcept
    {
        convolver_.convolve(input, kernel, out);

        for ( long_t i = 1; i < num_inputs; ++i )
        {
            convolver_.convolve_add(input +  i * in_image_len,
                                    kernel + i * kernel_len,
                                    out);
        }

        nonlinearity(out, bias);
    }
#endif

public:
    host_tensor<real,4> forward( host_tensor<float,4> in ) const override
    {
        host_tensor<real,4> out(output_shape);

#if defined(ZNN_USE_MKL_CONVOLUTION)

        std::vector<std::pair<long_t, long_t>> all(in_batch_size*num_outputs);

        tbb::concurrent_queue<std::pair<long_t, long_t>*> queue;

        for ( long_t n = 0, l = 0; n < in_batch_size; ++n )
            for ( long_t i = 0; i < num_outputs; ++i, ++l )
            {
                all[l].first  = n;
                all[l].second = i;
                queue.push(&all[l]);
            }

        auto fn = [&, this]()
            {
                host_tensor<real,2> scratch(this->out_image_size);

                std::pair<long_t,long_t>* which;

                while ( queue.try_pop(which) )
                {
                    this->do_single_output
                        ( in.data() + which->first * this->input_len,
                          kernels.data() + which->second * this->kernel_len * this->num_inputs,
                          out.data() + which->first * this->output_len + which->second * this->out_image_len,
			  (this->biases).data()[which->second],
                          scratch.data() );
                }

            };

	tbb::task_group tg;
        long_t num_tasks = std::thread::hardware_concurrency();
        num_tasks = std::min(num_tasks, in_batch_size*num_outputs);

        for ( long_t i = 0; i < num_tasks - 1; ++i )
        {
            tg.run(fn);
        }

        tg.run_and_wait(fn);

#else
        tbb::task_group tg;

        for ( long_t n = 0; n < in_batch_size; ++n )
        {
            for ( long_t i = 0; i < num_outputs; ++i )
            {
                tg.run([&,n,i]() {
                        this->do_single_output
                            ( in.data() + n * input_len,
                              kernels.data() + i * kernel_len * num_inputs,
                              out.data() + n * output_len + i * out_image_len,
                              biases.data()[i] );
                    });
            }
        }

        tg.wait();
#endif

        return out;
    }


    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }


};

}}}} // namespace znn::fwd::host::twod
