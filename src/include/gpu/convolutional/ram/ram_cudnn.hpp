#pragma once

#include "in_out_split_cudnn.hpp"
#include "batch_split_cudnn.hpp"
#include "../../../cpu/host_layer.hpp"
#include "../../../cpu/convolutional/base.hpp"
#include "../padd_pruned_cufft_native.hpp"

namespace znn { namespace fwd { namespace gpu {

template<typename Native>
class ram_cudnn_convolutional_layer
    : public cpu::cpu_convolutional_layer_base
    , public cpu::host_layer
{
private:
    std::unique_ptr<batch_split_cudnn_convolutional_layer<Native>>  impl1_;
    std::unique_ptr<in_out_split_cudnn_convolutional_layer<Native>> impl2_;

private:
    template<class L>
    host_array<real> forward1( host_array<real> in, L const * const l ) const
    {
        host_array<real> out = get_array<real>(total_output_len);

        real* outp = out.get();
        real* inp  = in.get();

        auto workspace = get_device_array<char>(l->workspace_size());

        l->forward(inp, outp, kernels.get(), biases.get(), workspace.get());

        return out;
    }

    template<class L>
    host_array<real> forward2( host_array<real> in, L const * const l ) const
    {
        host_array<real> out = get_array<real>(total_output_len);

        real* outp = out.get();
        real* inp  = in.get();

        auto workspace = get_device_array<char>(l->workspace_size());

        for ( long_t i = 0; i < batch_size; ++i )
        {
            l->forward(inp, outp, kernels.get(), biases.get(), workspace.get());
            inp  += input_len ;
            outp += output_len;
        }

        return out;
    }


public:
    host_array<real> forward( host_array<real> in ) const override
    {
        if ( impl1_ )
        {
            return forward1(std::move(in), impl1_.get());
        }
        else
        {
            return forward2(std::move(in), impl2_.get());
        }
    }

public:
    ram_cudnn_convolutional_layer( long_t n,
                                   long_t fin,
                                   long_t fout,
                                   vec3i const & is,
                                   vec3i const & ks,
                                   real* km = nullptr,
                                   real* bs = nullptr )
        : cpu::cpu_convolutional_layer_base( n, fin, fout, is, ks, km, bs )
    {
        long_t max_elements = 1000000000;
        long_t total_len    = total_input_len + total_output_len;

        if ( total_len < max_elements )
        {
            long_t n_chunk = max_elements / total_len;
            n_chunk = n_chunk > n ? n : n_chunk;

            STRONG_ASSERT(n_chunk>0);

            std::cout << "LAYER<batch_split>: " << n << ' '
                      << fin << ' ' << fout << ' '
                      << is << ' ' << ks << '\n'
                      << "  BREAKS INTO: " << n_chunk << "\n";

            impl1_ = std::unique_ptr<batch_split_cudnn_convolutional_layer<Native>>
                ( new batch_split_cudnn_convolutional_layer<Native>(
                    n, n_chunk, fin, fout, is, ks ));
        }
        else
        {
            long_t fin_chunk = 700*700*700 / in_image_len;
            fin_chunk = fin_chunk > fin ? fin : fin_chunk;

            long_t fout_chunk = 700*700*700 / out_image_len;
            fout_chunk = fout_chunk > fout ? fout : fout_chunk;

            STRONG_ASSERT(fin_chunk>0);
            STRONG_ASSERT(fout_chunk>0);

            std::cout << "LAYER<in_out_split>: " << fin << ' ' << fout << ' '
                      << is << ' ' << ks << '\n'
                      << "  BREAKS INTO: " << fin_chunk << ' '
                      << fout_chunk << "\n";

            impl2_ = std::unique_ptr<in_out_split_cudnn_convolutional_layer<Native>>
                ( new in_out_split_cudnn_convolutional_layer<Native>(
                    fin, fin_chunk, fout, fout_chunk, is, ks ));
        }
    }

};

typedef native_cudnn_convolutional_layer gemm_it;
typedef padded_pruned_cufft_native_convolutional_layer fft_it;

}}} // namespace znn::fwd::gpu
