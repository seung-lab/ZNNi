#pragma once

#include "znn/log.hpp"
#include "znn/host/v1/host_layer.hpp"
#include "znn/host/v1/conv_data.hpp"
#include "znn/device/ram/in_out_split_conv.hpp"
#include "znn/device/ram/detail/native_cudnn.hpp"
#include "znn/device/ram/detail/native_fft.hpp"

namespace znn { namespace fwd { namespace device { namespace ram {

template<typename Conv>
class ram_conv
    : public conv_layer<host::v1::host_layer>
    , public host::v1::conv_data
{
private:
    std::unique_ptr<in_out_split_conv<Conv>> impl_;

public:
    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

    host_tensor<float,5> forward( host_tensor<float,5> in ) const override
    {
        host_tensor<float,5> out(output_shape);
        device_array<char>   ws (impl_->workspace_size());

        float* outp = out.data();
        float* inp  = in.data();

        for ( long_t i = 0; i < batch_size; ++i )
        {
            impl_->forward(inp, outp,
                           kernels.data(), biases.data(), ws.data());

            inp  += input_len ;
            outp += output_len;
        }

        return out;
    }
public:
    ram_conv( long_t n, long_t fin, long_t fout,
              vec3i const & is, vec3i const & ks,
              float * km = nullptr, float* bs = nullptr )
        : conv_layer<host::v1::host_layer>(n,fin,fout,is,ks)
        , host::v1::conv_data(fin,fout,ks,km,bs)
    {

        long_t fin_chunk = 700*700*700 / in_image_len;
        //fin_chunk = 2;
        fin_chunk = fin_chunk > fin ? fin : fin_chunk;

        long_t fout_chunk = 700*700*700 / out_image_len;
        //fout_chunk = 2;
        fout_chunk = fout_chunk > fout ? fout : fout_chunk;

        STRONG_ASSERT(fin_chunk>0);
        STRONG_ASSERT(fout_chunk>0);

        // LOG(ram_conv) << "LAYER<in_out_split>: " << fin << ' ' << fout << ' '
        //               << is << ' ' << ks << '\n'
        //               << "  BREAKS INTO: " << fin_chunk << ' '
        //               << fout_chunk << "\n";

        impl_ = make_unique<in_out_split_conv<Conv>>
            (fin, fin_chunk, fout, fout_chunk, is, ks );

    }
};

typedef native_cudnn_conv gemm_it;
typedef native_fft_conv   fft_it;

}}}} // namespace znn::fwd::device::ram
