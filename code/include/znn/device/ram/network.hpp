#pragma once

#include "znn/host/v1/mfp.hpp"
#include "znn/device/ram/ram_conv.hpp"
#include "znn/device/tail/network.hpp"

namespace znn { namespace fwd { namespace device { namespace ram {

template<class T>
struct conv_traits;

template<>
struct conv_traits<gemm_it>
{
    typedef device::tail::cudnn_conv tail_type;
};

template<>
struct conv_traits<fft_it>
{
    typedef device::tail::fft_conv tail_type;
};

template<typename Conv>
class network
{
private:
    typedef typename conv_traits<Conv>::tail_type tail_type;

private:
    std::vector<std::unique_ptr<host::v1::host_layer>> stage1_;
    std::unique_ptr<device::tail::network<tail_type>>  stage2_;

    vec5i  in_shape_  ;
    vec5i  out_shape_ ;


    long_t memory_required_ = 0;

public:
    long_t memory_required() const
    {
        return memory_required_;
    }

    template<typename ND>
    network( ND const & nd, long_t cutoff )
        : stage1_(cutoff)
        , in_shape_(nd.in_shape())
        , out_shape_(nd.out_shape())
    {
        auto net = nd;

        for ( long_t i = 0; i < cutoff; ++i )
        {
            auto l = net.layers()[0];
            if ( l.type == layer_type::convolutional )
            {
                stage1_[i]
                    = make_unique<device::ram::ram_conv<Conv>>
                    (l.batch_size,
                     l.num_inputs,
                     l.num_outputs,
                     l.in_image_size,
                     l.k_or_w_size,
                     l.hkernels->data(),
                     l.hbiases->data());
            }
            else
            {
                stage1_[i]
                    = make_unique<host::v1::mfp>
                    (l.batch_size,
                     l.num_inputs,
                     l.in_image_size,
                     l.k_or_w_size);
            }
            memory_required_ = std::max(memory_required_,
                                        stage1_[i]->working_memory());

            net = net.tail();
        }

        for ( auto & l: net.layers() )
        {
            std::cout << l.batch_size << ' '
                      << l.num_inputs << ' '
                      << l.num_outputs << ' '
                      << l.in_image_size << '\n';
        }

        stage2_ = device::tail::network<
            typename conv_traits<Conv>::tail_type>::get(net);

        if ( !stage2_ )
        {
            throw std::logic_error("no feasable network");
        }

        memory_required_ = std::max(memory_required_,
                                    stage2_->working_memory());

    }

    vec5i const & in_shape() const
    {
        return in_shape_;
    }

    vec5i const & out_shape() const
    {
        return out_shape_;
    }

    host_tensor<real,5> forward( host_tensor<real,5> in ) const
    {
        for ( auto const & l: stage1_ )
        {
            in = l->forward(std::move(in));
        }
        return stage2_->forward(std::move(in));
    }


};


}}}} // namespace znn::fwd::device::ram
