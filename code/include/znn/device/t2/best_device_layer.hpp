#pragma once

#include "znn/device/v2/cudnn_conv.hpp"
#include "znn/device/v2/fft_conv.hpp"
#include "znn/device/v2/cudnn_mfp.hpp"
#include "znn/device/v2/network_data.hpp"

namespace znn { namespace fwd { namespace device { namespace v2 {

inline std::unique_ptr<device_layer>
get_best_device_layer( network_data::layer_data const & l )
{
    if ( l.type == layer_type::convolutional )
    {
        std::cout << "Trying CONV" << std::endl;

        std::unique_ptr<device_layer> ret = nullptr;
        double best = std::numeric_limits<double>::max();

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
            auto b = x->benchmark();

            if ( b < best )
            {
                ret = std::move(x);
                best = b;
            }
        }
        catch (...)
        {
        }

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
            auto b = x->benchmark();

            if ( b < best )
            {
                ret = std::move(x);
                best = b;
            }
        }
        catch (...)
        {
        }

        return ret;
    }
    else
    {
        try
        {
            std::cout << "Trying MFP" << std::endl;
            std::unique_ptr<device_layer> x
                = make_unique<cudnn_mfp>(l.batch_size,
                                         l.num_inputs,
                                         l.in_image_size,
                                         l.k_or_w_size);
            std::cout << "Trying MFP " << x.get() << std::endl;
            x->benchmark();
            return x;
        }
        catch (...)
        {
            return nullptr;
        }
    }
}


}}}} // namespace znn::fwd::device::v2
