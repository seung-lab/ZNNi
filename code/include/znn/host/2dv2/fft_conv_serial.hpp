#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/2dv2/host_layer.hpp"
#include "znn/host/2dv2/conv_data.hpp"
#include "znn/host/common/fft2d/fft.hpp"
#include "znn/host/common/thread_pin.hpp"
#include "znn/host/common/complex_mad.hpp"
#include "znn/host/common/complex_split_mad.hpp"

#include <atomic>
#include <thread>
#include <tbb/tbb.h>


namespace znn { namespace fwd { namespace host { namespace twod {

class fft_conv2d_serial
    : public conv_layer2d<host_layer2d>
    , public conv_data2d
{
private:
    padded_pruned_fft2d_transformer* fft_;
    long_t pcelements_;
    host_tensor<complex,3> kffts_;

public:
    fft_conv2d_serial( long_t n, long_t fin, long_t fout,
                       vec2i const & is, vec2i const & ks,
                       real * km = nullptr, real* bs = nullptr )
        : conv_layer2d<host_layer2d>(n,fin,fout,is,ks)
        , conv_data2d(fin,fout,ks,km,bs)
        , fft_(padded_pruned_fft2d_plans.get(is,ks))
        , pcelements_(fft_->p_len())
        , kffts_(fout,fin,pcelements_)
    {
        long_t total = fin * fout;
        long_t zeros = fft_->kernel_scratch_memory() - fft_->kernel_memory();

        auto fn = [&]( long_t i )
            {
                host_array<real> scratch(fft_->kernel_scratch_elements());
                real const * kernel = kernels.data() + i*kernel_len;
                complex * ckernel = kffts_.data() + i*pcelements_;

                std::memcpy( scratch.data(), kernel, fft_->kernel_memory());
                std::memset( scratch.data() + fft_->kernel_elements(), 0, zeros );

                fft_->forward_kernel( scratch.data(), ckernel );
            };

        tbb::parallel_for( static_cast<long_t>(0), total, fn );
    }


private:
    void do_input_fft( real* in, complex* out ) const noexcept
    {
        fft_->forward_image(in,out);
    }

    void do_input_fft_padded( real* in,
                              complex* out,
                              real* tmp) const noexcept
    {
        std::memcpy( tmp, in, fft_->image_memory());

        // append zeros
        long_t zero_bytes = fft_->image_scratch_memory() - fft_->image_memory();

        std::memset( tmp + fft_->image_elements(), 0, zero_bytes );

        fft_->forward_image(tmp,out);
    }

    void do_output_ifft( complex* in, real* out,
                         real bias, real* out_scratch ) const noexcept
    {
        fft_->backward(in,out_scratch);
        real scale = fft_->get_scale();

        long_t off = fft_->result_offset();

        for ( long_t i = 0; i < out_image_len; ++i )
        {
            out[i] = out_scratch[i+off] / scale + bias;
        }
    }

    void mul_to( complex const * a, complex const * b, complex * r,
                 long_t n ) const noexcept
    {
        complex_mst(a,b,r,n);
        // const real* ap = reinterpret_cast<real const *>(a);
        // const real* bp = reinterpret_cast<real const *>(b);
        // real* rp = reinterpret_cast<real *>(r);

        // complex_split_mst(ap, ap+pcelements_,
        //                   bp, bp+pcelements_,
        //                   rp, rp+pcelements_,
        //                   static_cast<size_t>(n));
    }

    void mul_add_to( complex const * a, complex const * b, complex * r,
                     long_t n ) const noexcept
    {
        complex_mad(a,b,r,n);

        // const real* ap = reinterpret_cast<real const *>(a);
        // const real* bp = reinterpret_cast<real const *>(b);
        // real* rp = reinterpret_cast<real *>(r);

        // complex_split_mad(ap, ap+pcelements_,
        //                   bp, bp+pcelements_,
        //                   rp, rp+pcelements_,
        //                   static_cast<size_t>(n));
    }


private:
    void transform_inputs( real* in, complex* itransforms,
                           real* scratch ) const noexcept
    {
        long_t relements = fft_->image_elements();
        //long_t celements = fft_->transform_elements();

        long_t total = num_inputs;

        if ( !this->fft_->needs_padding() )
        {
            for ( long_t i = 0; i < total; ++i )
                this->do_input_fft(in + relements * i,
                                   itransforms + pcelements_ * i);
        }
        else
        {
            for ( long_t i = 0; i < total; ++i )
                this->do_input_fft_padded(in + relements * i,
                                          itransforms + pcelements_ * i,
                                          scratch);
        }
    }

    void collect_outputs( complex* in, real* out,
                          complex* sum, real* scratch ) const noexcept
    {
        long_t celements = fft_->transform_elements();

        for ( long_t o = 0; o < num_outputs; ++o )
        {
            mul_to(in, kffts_[o][0].data(), sum, celements);

            for ( long_t i = 1; i < num_inputs; ++i )
            {
                mul_add_to(in + i * pcelements_, kffts_[o][i].data(),
                           sum, celements);
            }

            do_output_ifft( sum, out + o * out_image_len,
                            biases.data()[o], scratch );
        }
    }



public:
    void forward( real* in, real* out, void* ws ) const override
    {
        complex * itransforms = reinterpret_cast<complex*>(ws);
        complex * sum = itransforms + pcelements_ * num_inputs;
        real * scratch
            = reinterpret_cast<real*>(sum + pcelements_);

        for ( long_t i = 0; i < batch_size; ++i )
        {
            transform_inputs(in, itransforms, scratch);
            collect_outputs(itransforms, out, sum, scratch);

            in  += input_len ;
            out += output_len;
        }
    }

    long_t resident_memory() const override
    {
        return kernels_memory + bias_memory +
            num_inputs * num_outputs * pcelements_
            * sizeof(complex);
    }

    long_t working_memory() const override
    {
        return 0;
    }

    virtual long_t workspace_size() const
    {
        return (num_inputs + 2) * pcelements_ * sizeof(complex);
    }

};


}}}} // namespace znn::fwd::host::twod
