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
    long_t n_singles_;

public:
    host_array<real> forward( host_array<real> in ) const override
    {
        host_array<real> out = get_array<real>(total_output_len);

        real* outp = out.get();
        real* inp  = in.get();

        auto workspace = get_device_array<char>(single_->workspace_size());

        for ( long_t i = 0; i < n_singles_; ++i )
        {
            full_->forward(inp, outp, kernels.get(),
                           biases.get(), workspace.get());

            inp  += input_len ;
            outp += output_len;
        }

        return out;
    }

public:
    in_split_cudnn_convolutional_layer( handle_t& handle,
                                        long_t fin,
                                        long_t fin_chunk,
                                        long_t fout,
                                        vec3i const & is,
                                        vec3i const & ks )
        : gpuram_layer_base(1, fin, fout, is, ks)
    {
        n_full_ = fin / fin_chunk;

        full_ = std::unique_ptr<native_cudnn_convolutional_layer>
            ( new native_cudnn_convolutional_layer
              (handle, fin_chunk, fout, is, ks));

        workspace_size_ = full_->workspace_size();

        if ( fin % fin_chunk )
        {
            partial_ = std::unique_ptr<native_cudnn_convolutional_layer>
                ( new native_cudnn_convolutional_layer
                  (handle, fin%fin_chunk, fout, is, ks));

            workspace_size_ = std::max(workspace_size_,
                                       partial_->workspace_size());
        }
    }

};

}}} // namespace znn::fwd::gpu
