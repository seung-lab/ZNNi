#pragma once

#include "in_out_split_cudnn.hpp"
#include "../../../cpu/host_layer.hpp"
#include "../../../cpu/convolutional/base.hpp"

namespace znn { namespace fwd { namespace gpu {

class ram_cudnn_convolutional_layer
    : public cpu::cpu_convolutional_layer_base
    , public cpu::host_layer
{
private:
    std::unique_ptr<in_out_split_cudnn_convolutional_layer> single_;

public:
    host_array<real> forward( host_array<real> in ) const override
    {
        host_array<real> out = get_array<real>(total_output_len);

        real* outp = out.get();
        real* inp  = in.get();

        auto workspace = get_device_array<char>(single_->workspace_size());

        for ( long_t i = 0; i < batch_size; ++i )
        {
            single_->forward(inp, outp, kernels.get(),
                             biases.get(), workspace.get());

            inp  += input_len ;
            outp += output_len;
        }

        return out;
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
        long_t fin_chunk = 700*700*700 / in_image_len;
        fin_chunk = fin_chunk > fin ? fin : fin_chunk;

        long_t fout_chunk = 700*700*700 / out_image_len;
        fout_chunk = fout_chunk > fout ? fout : fout_chunk;

        STRONG_ASSERT(fin_chunk>0);
        STRONG_ASSERT(fout_chunk>0);

        std::cout << "LAYER: " << fin << ' ' << fout << ' '
                  << is << ' ' << ks << '\n'
                  << "  BREAKS INTO: " << fin_chunk << ' '
                  << fout_chunk << "\n";

        single_ = std::unique_ptr<in_out_split_cudnn_convolutional_layer>
            ( new in_out_split_cudnn_convolutional_layer(
                fin, fin_chunk, fout, fout_chunk, is, ks ));
    }

};

}}} // namespace znn::fwd::gpu
