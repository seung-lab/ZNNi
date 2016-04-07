#pragma once

#include "znn/log.hpp"
#include "znn/util/network.hpp"
#include "znn/fusion/device/cudnn_conv.hpp"
#include "znn/fusion/device/cudnn_conv_precomp.hpp"
#include "znn/fusion/device/fft_conv.hpp"
#include "znn/fusion/device/mfp.hpp"

#include <sstream>

namespace znn { namespace fwd { namespace device { namespace fusion {

class sub_network
{
private:
    std::vector<std::unique_ptr<device_layer>> layers;
    long_t batch_size_;
    long_t delta_in_  ;
    long_t delta_out_ ;
    vec5i  in_shape_  ;
    vec5i  out_shape_ ;

    double estimate_;

public:
    long_t resident_memory() const
    {
        long_t r = 0;
        for ( auto & l: layers )
        {
            r += l->resident_memory();
        }
        return r;
    }

    long_t working_memory() const
    {
        long_t r = 0;
        for ( auto & l: layers )
        {
            r = std::max(r, l->working_memory());
        }
        return r;
    }


private:
    template<typename T>
    std::unique_ptr<device_layer>
    best_layer( T const & l, bool b )
    {
        if ( l.type == layer_type::convolutional )
        {
            std::unique_ptr<device_layer> ret = nullptr;
            double best = std::numeric_limits<double>::max();

            LOG(best_layer) << "find best conv layer: "
                            << l.batch_size << ' '
                            << l.num_inputs << "->"
                            << l.num_outputs << ' '
                            << l.in_image_size << ' '
                            << l.k_or_w_size;

            LOG(best_layer) << "trying cudnn_conv_precomp";

            // try cudnn
            try
            {
                auto x = make_unique<cudnn_conv_precomp>(l.batch_size,
                                                         l.num_inputs,
                                                         l.num_outputs,
                                                         l.in_image_size,
                                                         l.k_or_w_size,
                                                         l.kernels,
                                                         l.biases);
                auto b = x->estimate_runtime();

                LOG(best_layer) << "cudnn_conv_precomp time: " << b;

                if ( b < best )
                {
                    LOG(best_layer) << "cudnn_conv_precomp (best) time: " << b;
                    ret = std::move(x);
                    best = b;
                }
            }
            catch ( std::exception & e )
            {
                LOG(best_layer) << "cudnn_conv_precomp failed: " << e.what();
            }

            LOG(best_layer) << "trying fft_conv";

            // try fft
            try
            {
                auto x = make_unique<fft_conv>(l.batch_size,
                                               l.num_inputs,
                                               l.num_outputs,
                                               l.in_image_size,
                                               l.k_or_w_size,
                                               l.kernels,
                                               l.biases);
                auto b = x->estimate_runtime();

                LOG(best_layer) << "fft_conv time: " << b;

                if ( b < best )
                {
                    LOG(best_layer) << "fft_conv (best) time: " << b;
                    ret = std::move(x);
                    best = b;
                }
            }
            catch ( std::exception & e )
            {
                LOG(best_layer) << "cudnn_conv_precomp failed: " << e.what();
            }

            // if ( ret )
            // {
            //     estimate_ += best;
            //     return ret;
            // }

            // if ( !b )
            // {
            //     return ret;
            // }

            LOG(best_layer) << "trying cudnn_conv";

            // try cudnn
            try
            {
                auto x = make_unique<cudnn_conv>(l.batch_size,
                                                 l.num_inputs,
                                                 l.num_outputs,
                                                 l.in_image_size,
                                                 l.k_or_w_size,
                                                 l.kernels,
                                                 l.biases);
                auto b = x->estimate_runtime();

                LOG(best_layer) << "cudnn_conv time: " << b;

                if ( b < best )
                {
                    LOG(best_layer) << "cudnn_conv (best) time: " << b;
                    ret = std::move(x);
                    best = b;
                }
            }
            catch ( std::exception & e )
            {
                LOG(best_layer) << "cudnn_conv failed: " << e.what();
            }

            estimate_ += best;
            return ret;
        }
        else
        {
            LOG(best_layer) << "find best mfp layer: "
                            << l.batch_size << ' '
                            << l.num_inputs << "->"
                            << l.num_outputs << ' '
                            << l.in_image_size << ' '
                            << l.k_or_w_size;

            try
            {
                LOG(best_layer) << "trying mfp";

                std::unique_ptr<device_layer> x
                    = make_unique<mfp>(l.batch_size,
                                       l.num_inputs,
                                       l.in_image_size,
                                       l.k_or_w_size);

                auto b = x->estimate_runtime();

                LOG(best_layer) << "trying mfp time: " << b;

                estimate_ += b;
                return x;
            }
            catch ( std::exception & e )
            {
                LOG(best_layer) << "mfp failed: " << e.what();
                return nullptr;
            }
        }
    }


public:
    template<typename ND>
    sub_network( ND const & nd, bool b )
        : layers()
        , batch_size_(nd.batch_size())
        , delta_in_(nd.delta_in())
        , delta_out_(nd.delta_out())
        , in_shape_(nd.in_shape())
        , out_shape_(nd.out_shape())
        , estimate_(0)
    {
        //  Check for very large input/outputs

        if ( ((nd.delta_in() + nd.delta_out()) / 1024 / 1024 / 1024) > 12 )
        {
            throw std::logic_error("no feasable sub_network");
        }

        for ( auto & l: nd.layers() )
        {
            auto best = best_layer(l, b);
            if ( best )
            {
                layers.push_back(std::move(best));
            }
            else
            {
                throw std::logic_error("no feasable sub_network");
            }
        }
    }

public:
    void forward( float const * in, float * out ) const
    {
        device_tensor<float,5> din(in_shape_);
        din.load(in, from_host);

        for ( auto & l: layers )
        {
            din = l->forward(std::move(din));
        }

        din.store(out, to_host);
    }

    double estimate_runtime() const
    {
        zi::wall_timer wt;
        host_tensor<float,5> out(out_shape_);
        host_tensor<float,5> in(rand_init,in_shape_);
        wt.reset();
        forward(in.data(),out.data());
        return wt.elapsed<double>();
    }

    long_t batch_size() const
    {
        return batch_size_;
    }

    long_t delta_in() const
    {
        return delta_in_;
    }

    long_t delta_out() const
    {
        return delta_out_;
    }

    vec5i const & in_shape() const
    {
        return in_shape_;
    }

    vec5i const & out_shape() const
    {
        return out_shape_;
    }

    template<typename ND>
    static std::unique_ptr<sub_network> get( ND const & nd, bool b )
    {
        std::unique_ptr<sub_network> ret;
        try
        {
            ret = make_unique<sub_network>(nd, b);
        }
        catch ( std::exception & e )
        {
            LOG(sub_network) << "failed to get sub_network: " << e.what();
        }

        return ret;
    }

    std::string name() const
    {
        std::ostringstream oss;
        oss << "layers: ";
        for ( auto & l: layers )
        {
            oss << ' ' << l->name();
        }
        return oss.str();
    }
};


}}}} // namespace znn::fwd::device::fusion
