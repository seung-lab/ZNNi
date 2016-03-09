#pragma once

#include "cudnn.hpp"
#include "padded_pruned_cufft.hpp"

#include <zi/time.hpp>
#include <limits>

namespace znn { namespace fwd { namespace gpu {


class faster_convolutional_layer
    : public convolutional_layer_base
    , public device_layer
{
private:
    double best_time;
    std::unique_ptr<device_layer> impl_;

public:
    long_t permanent_memory_required() const override
    {
        return impl_ ? impl_->permanent_memory_required() : -1;
    }

    long_t working_memory_required() const override
    {
        return impl_ ? impl_->working_memory_required() : -1;
    }

    device_array<float> forward( device_array<float> in ) const override
    {
        return impl_->forward(std::move(in));
    }

private:
    template<typename L>
    bool attempt( long_t n, long_t fin, long_t fout,
                  vec3i const & is, vec3i const & ks,
                  float* km, float* bs )
    {
        std::unique_ptr<device_layer> l(new L(n, fin, fout, is, ks, km, bs));

        if ( (l->working_memory_required() / 1024 / 1024) < 10000 )
        {
            zi::wall_timer wt;
            auto in = get_device_array<float>(total_input_len);
            wt.reset();
            auto out = l->forward(std::move(in));
            double t = wt.elapsed<double>();

            std::cout << "\ttook: " << t << " secs\n";

            if ( t < best_time )
            {
                best_time = t;
                impl_ = std::move(l);
                return true;
            }
            return false;
        }
        else
        {
            std::cout << "\tNo enough memory\n";
            return false;
        }
    }

public:
    faster_convolutional_layer( long_t n, long_t fin, long_t fout,
                                vec3i const & is, vec3i const & ks,
                                float* km = nullptr, float* bs = nullptr )
        : convolutional_layer_base(n,fin,fout,is,ks)
        , best_time(std::numeric_limits<double>::max())
    {
        std::cout << "Attempting to pick a better layer for:"
                  << "\n\t" << n << ' ' << fin << ' ' << fout
                  << "\n\t" << is << ' ' << ks << "\n";

        std::cout << "Trying CUDNN\n";
        this->template attempt<cudnn_convolutional_layer>
            (n, fin, fout, is, ks, km, bs);
        this->template attempt<padded_pruned_cufft_convolutional_layer>
            (n, fin, fout, is, ks, km, bs);

    }
};



}}} // namespace znn::fwd::gpu
