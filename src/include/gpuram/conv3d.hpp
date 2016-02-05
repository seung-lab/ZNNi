#pragma once

#include "conv/full_conv3d.hpp"
#include "conv/batch_split_conv3d.hpp"
#include "../cpu/cpu3d.hpp"

#define _MAX_ELEMENTS 500000000

namespace znn { namespace fwd { namespace gpu3dram {

class conv_layer: public cpu3d::cpu_layer
{
private:
    base_conv3d* impl_ = nullptr;

    real * kernel_data_ ;
    real * bias_data_   ;

    long_t total_out_elements;

public:

    real* kernel_data()
    {
        return kernel_data_;
    }

    real* bias_data()
    {
        return bias_data_;
    }

    ~conv_layer()
    {
        znn_free(kernel_data_);
        znn_free(bias_data_);
        if (impl_ ) delete impl_ ;
    }

    int in_memory() const override
    {
        return 0;
    }

    int out_memory() const override
    {
        return 0;
    }

    conv_layer( cudnnHandle_t& handle,
                int n,
                int fin,
                int fout,
                vec3i const & is,
                vec3i const & fs )
    {
        kernel_data_ = znn_malloc<real>(fs[0] * fs[1] * fs[2] * fin * fout);
        bias_data_   = znn_malloc<real>(fout);

        vec3i os = is - fs + vec3i::one;

        total_out_elements = os[0] * os[1] * os[2] * n * fout;

        long_t input_elements  = is[0] * is[1] * is[2];
        long_t output_elements = os[0] * os[1] * os[2];

        long_t elements = input_elements + output_elements;

        long_t n_input_elements  = is[0] * is[1] * is[2] * n;
        long_t n_output_elements = os[0] * os[1] * os[2] * n;

        long_t n_elements = n_input_elements + n_output_elements;

        if ( n_elements <= _MAX_ELEMENTS )
        {
            impl_ = new full_conv3d(handle, n, fin, fout, is, fs);
        }
        else if ( elements <= _MAX_ELEMENTS )
        {
            long_t n_batch = n_elements / _MAX_ELEMENTS;
            impl_ = new batch_split_conv3d( handle, n, n_batch,
                                            fin, fout, is, fs );
        } else
            DIE("Not supported");
        }
    }

    real * forward( real * in ) override
    {
        real * out = znn_malloc<real>(total_out_elements);
        impl_->forward(in, out, kernel_data_, bias_data_);
        znn_free(in);
        return out;
    }

};


}}} // namespace znn::fwd::gpu3dram
