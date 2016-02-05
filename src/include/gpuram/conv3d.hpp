#pragma once

#include "conv/full_conv3d.hpp"
#include "conv/batch_split_conv3d.hpp"
#include "conv/full_split_conv3d.hpp"
#include "../cpu/cpu3d.hpp"

#define _MAX_ELEMENTS 500000000

namespace znn { namespace fwd { namespace gpu3dram {

class conv_layer: public cpu3d::cpu_layer
{
private:
    base_conv3d* impl_ = nullptr;

    real * kernel_data_ ;
    real * bias_data_   ;

    long_t in_memory_ ;
    long_t out_memory_;

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

    long_t in_memory() const override
    {
        return in_memory_;
    }

    long_t out_memory() const override
    {
        return out_memory_;
    }

    conv_layer( cudnnHandle_t& handle,
                long_t n,
                long_t fin,
                long_t fout,
                vec3i const & is,
                vec3i const & fs )
    {
        kernel_data_ = znn_malloc<real>(fs[0] * fs[1] * fs[2] * fin * fout);
        bias_data_   = znn_malloc<real>(fout);

        vec3i os = is - fs + vec3i::one;

        total_out_elements = os[0] * os[1] * os[2] * n * fout;

        long_t input_elements  = is[0] * is[1] * is[2] * fin;
        long_t output_elements = os[0] * os[1] * os[2] * fout;

        long_t elements = input_elements + output_elements;

        long_t n_input_elements  = is[0] * is[1] * is[2] * fin  * n;
        long_t n_output_elements = os[0] * os[1] * os[2] * fout * n;

        in_memory_  = n_input_elements  * sizeof(float);
        out_memory_ = n_output_elements * sizeof(float);

        long_t n_elements = n_input_elements + n_output_elements;

        if ( n_elements <= _MAX_ELEMENTS )
        {
            std::cout << "WILL USE FULL_CONV3D" << std::endl;
            impl_ = new full_conv3d(handle, n, fin, fout, is, fs);
        }
        else if ( elements <= _MAX_ELEMENTS )
        {
            long_t n_batch = _MAX_ELEMENTS / elements;
            std::cout << "WILL USE BATCH_SPLIT_CONV3D: "
                      << n_batch << std::endl;
            impl_ = new batch_split_conv3d( handle, n, n_batch,
                                            fin, fout, is, fs );
        } else
        {
            long_t fin_chunk  = std::max(static_cast<long_t>(1),
                                         _MAX_ELEMENTS/2/(input_elements/fin) );
            long_t fout_chunk = std::max(static_cast<long_t>(1),
                                         _MAX_ELEMENTS/2/(output_elements/foutn));
            fin_chunk = std::min(fin_chunk,fin);
            fout_chunk = std::min(fout_chunk,fout);
            std::cout << "WILL USE FULL_SPLIT_CONV3D: "
                      << fin_chunk << ' ' << fout_chunk << std::endl;
            impl_ = new full_split_conv3d( handle, n,
                                           fin, fin_chunk,
                                           fout, fout_chunk,
                                           is, fs );
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
