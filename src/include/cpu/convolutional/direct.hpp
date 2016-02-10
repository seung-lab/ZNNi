#pragma once

#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../memory.hpp"
#include "../../layer.hpp"
#include "../host_layer.hpp"
#include "../utils/task_package.hpp"
#include "conv/convolver.hpp"
#include "base.hpp"

namespace znn { namespace fwd { namespace cpu {

class direct_convolutional_layer
    : public cpu_convolutional_layer_base
    , public host_layer
{
private:
    task_package &             handle_   ;
    std::unique_ptr<convolver> convolver_;

public:
    direct_convolutional_layer( task_package& handle,
                                long_t n, long_t fin, long_t fout,
                                vec3i const & is, vec3i const & ks,
                                real * km = nullptr, real* bs = nullptr )
        : cpu_convolutional_layer_base( n, fin, fout, is, ks, km, bs )
        , handle_(handle)
        , convolver_(new convolver(is,ks))
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
                           real* kernel
                           real* out,
                           real  bias,
                           void* stack) const noexcept
    {
        convolver_->convolve_add(input, kernel, out);

        real* tmp = reinterpret_cast<real*>(stack);

        for ( long_t i = 0; i < num_inputs; ++i )
        {
            convolver_->convolve_add(input +  i * in_image_len,
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
                           real* kernel,
                           real* out,
                           real  bias,
                           void* ) const noexcept
    {
        convolver_->convolve(input, kernel, out);

        for ( long_t i = 1; i < num_inputs; ++i )
        {
            convolver_->convolve_add(input +  i * in_image_len,
                                     kernel + i * kernel_len,
                                     out);
        }

        nonlinearity(out, bias);
    }
#endif

public:
    host_array<real> forward( host_array<real> in ) const override
    {
        host_array<real> out = get_array<real>(total_output_len);

        for ( long_t n = 0; n < batch_size; ++n )
        {
            for ( long_t i = 0; i < num_outputs; ++i )
            {
                real* first_kernel
                    = kernels.get() + i * kernel_len * num_inputs;

                handle_.add_task( &direct_convolutional_layer::do_single_output,
                                  this,
                                  in.get() + n * input_len,
                                  first_kernel,
                                  out.get() + n * output_len + i * out_image_len,
                                  biases.get()[i] );

            }
        }

#if defined(ZNN_USE_MKL_CONVOLUTION)
        handle_.execute(out_image_len*sizeof(real));
#else
        handle_.execute();
#endif

        return out;
    }


};

}}} // namespace znn::fwd::cpu
