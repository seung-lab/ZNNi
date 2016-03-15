#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/host_layer.hpp"
#include "znn/host/v1/conv_data.hpp"
#include "znn/host/common/conv/convolver.hpp"

#include <tbb/tbb.h>

namespace znn { namespace fwd { namespace host { namespace v1 {

class direct_conv
    : public conv_layer<host_layer>
    , public conv_data
{
private:
    convolver convolver_;

public:
    direct_conv( long_t n, long_t fin, long_t fout,
                 vec3i const & is, vec3i const & ks,
                 float * km = nullptr, float* bs = nullptr )
        : conv_layer<host_layer>(n,fin,fout,is,ks)
        , conv_data(fin,fout,ks,km,bs)
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
                           real const * kernel
                           real* out,
                           real  bias,
                           void* stack) const noexcept
    {
        convolver_.convolve_add(input, kernel, out);

        real* tmp = reinterpret_cast<real*>(stack);

        for ( long_t i = 0; i < num_inputs; ++i )
        {
            convolver_.convolve_add(input +  i * in_image_len,
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
    host_tensor<real,5> forward( host_tensor<float,5> in ) const override
    {
        host_tensor<real,5> out(output_shape);

#if defined(ZNN_USE_MKL_CONVOLUTION)

        UNIMPLEMENTED();

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

}}}} // namespace znn::fwd::host::v1
