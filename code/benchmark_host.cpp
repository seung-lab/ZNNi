#include "znn/util/network.hpp"
#include "znn/host/v1/mfp.hpp"
#include "znn/host/v1/fft_conv.hpp"
#include "znn/host/v1/dp_fft_conv.hpp"
#include "znn/host/v1/direct_conv.hpp"

#include <zi/time.hpp>

#include <string>
#include <fstream>
#include <sstream>

// RUN THE FOLLOWING
// ./benchmark_host m36 8
// ./benchmark_host m56 8
// ./benchmark_host m76 8
// ./benchmark_host m96 8

using namespace znn::fwd;

std::string net_name;
vec3i       os;
long_t      max_memory = static_cast<long_t>(240) * 1024 * 1024 * 1024; // GB

inline double benchmark_mfp_layer( znni_network::znni_layer const & layer,
                                   long_t layer_num,
                                   long_t rounds,
                                   std::ofstream & rout )
{
    std::ostringstream oss;

    oss << "## " << net_name << " :: " << os << " :: "
        << layer_num << " :: "
        << " MFP";

    std::string prefix = oss.str();

    double best = std::numeric_limits<double>::max();

    // data parallel
    {
        host::v1::mfp l(layer.batch_size,
                        layer.descriptor.num_inputs,
                        layer.in_size,
                        layer.descriptor.k_or_w_size);

        rout << prefix << " requires "
             << (l.total_memory() / 1024 / 1024 / 1024) << "GB" << std::endl;

        if ( l.total_memory() < max_memory )
        {
            zi::wall_timer wt;
            double total = 0;

            for ( long_t i = 0; i < rounds; ++i )
            {
                auto in = layer.get_random_sample();
                wt.reset();
                auto out = l.forward(std::move(in));
                double t = wt.elapsed<double>();
                out.reset();

                total += t;
                rout << "[layer_measurement] " << net_name << " :: " << os << " :: "
                     << layer_num << " :: MFP :: " << t << std::endl;
            }

            total /= rounds;
            best = total;

            rout << "[layer_average_measurement] " << net_name << " :: " << os << " :: "
                 << layer_num << " :: MFP :: " << total << std::endl;
        }
    }

    return best;
}


inline double benchmark_conv_layer( znni_network::znni_layer const & layer,
                                    long_t layer_num,
                                    long_t rounds,
                                    std::ofstream & rout )
{
    std::ostringstream oss;

    oss << "## " << net_name << " :: " << os << " :: "
        << layer_num << " :: "
        << " CONVOLUTIONAL";

    std::string prefix = oss.str();

    double best = std::numeric_limits<double>::max();

    // data parallel
    {
        host::v1::dp_fft_conv l(layer.batch_size,
                                layer.descriptor.num_inputs,
                                layer.descriptor.num_outputs,
                                layer.in_size,
                                layer.descriptor.k_or_w_size,
                                layer.random_kernels().data(),
                                layer.random_biases().data());

        rout << prefix << " DATA PARALLEL FFT requires "
             << (l.total_memory() / 1024 / 1024 / 1024) << "GB" << std::endl;

        if ( l.total_memory() < max_memory )
        {
            zi::wall_timer wt;
            double total = 0;

            for ( long_t i = 0; i < rounds; ++i )
            {
                auto in = layer.get_random_sample();
                wt.reset();
                auto out = l.forward(std::move(in));
                double t = wt.elapsed<double>();
                out.reset();

                total += t;
                rout << "[layer_measurement] " << net_name << " :: " << os << " :: "
                     << layer_num << " :: DP_FFT :: " << t << std::endl;
            }

            total /= rounds;
            best = total;

            rout << "[layer_average_measurement] " << net_name << " :: " << os << " :: "
                 << layer_num << " :: DP_FFT :: " << total << std::endl;
        }
    }

    // regular
    {
        host::v1::fft_conv l(layer.batch_size,
                             layer.descriptor.num_inputs,
                             layer.descriptor.num_outputs,
                             layer.in_size,
                             layer.descriptor.k_or_w_size,
                             layer.random_kernels().data(),
                             layer.random_biases().data());

        rout << prefix << " DATA FFT requires "
             << (l.total_memory() / 1024 / 1024 / 1024) << "GB" << std::endl;

        if ( l.total_memory() < max_memory )
        {
            zi::wall_timer wt;
            double total = 0;

            for ( long_t i = 0; i < rounds; ++i )
            {
                auto in = layer.get_random_sample();
                wt.reset();
                auto out = l.forward(std::move(in));
                double t = wt.elapsed<double>();
                out.reset();

                total += t;
                rout << "[layer_measurement] " << net_name << " :: " << os << " :: "
                     << layer_num << " :: FFT :: " << t << std::endl;
            }

            total /= rounds;
            best = std::min(total,best);

            rout << "[layer_average_measurement] " << net_name << " :: " << os << " :: "
                 << layer_num << " :: DP_FFT :: " << total << std::endl;

        }
    }

    rout << "[layer_best_measurement] " << net_name << " :: " << os << " :: "
         << layer_num << " :: DP_FFT :: " << best << std::endl;


    return best;
}

inline void benchmark_network( network_descriptor & ndesc,
                               long_t rounds,
                               std::ofstream & rout )
{
    znni_network net(ndesc, 1, os);

    rout << "## " << net_name << " :: starting benchmark for output size "
         << os << std::endl;

    std::vector<std::unique_ptr<host::v1::host_layer>> layers;

    long_t lnum = 0;

    long_t rm = 0;
    long_t wm = 0;

    for ( auto & l: net.layers() )
    {
        if ( l.descriptor.type == layer_type::convolutional )
        {
            if ( lnum == 0 )
            {
                layers.push_back(std::unique_ptr<host::v1::host_layer>
                                 (new host::v1::dp_fft_conv
                                  (l.batch_size,
                                   l.descriptor.num_inputs,
                                   l.descriptor.num_outputs,
                                   l.in_size,
                                   l.descriptor.k_or_w_size,
                                   l.random_kernels().data(),
                                   l.random_biases().data())));
            }
            else
            {
                layers.push_back(std::unique_ptr<host::v1::host_layer>
                                 (new host::v1::fft_conv
                                  (l.batch_size,
                                   l.descriptor.num_inputs,
                                   l.descriptor.num_outputs,
                                   l.in_size,
                                   l.descriptor.k_or_w_size,
                                   l.random_kernels().data(),
                                   l.random_biases().data())));
            }
        }
        else
        {
            layers.push_back(std::unique_ptr<host::v1::host_layer>
                             (new host::v1::mfp
                              (l.batch_size,
                               l.descriptor.num_inputs,
                               l.in_size,
                               l.descriptor.k_or_w_size)));
        }

        ++lnum;

        rm += layers.back()->resident_memory();
        wm = std::max(wm, layers.back()->working_memory());
    }

    rout << "[network_requirements] " << net_name
         << " :: " << os << " :: "
         << "RM: " << rm/1024/1024/1024
         << " GB WM: " << wm/1024/1024/1024 << " GB" << std::endl;

    if ( rm + wm < max_memory )
    {
        double total = 0;
        zi::wall_timer wtn;
        zi::wall_timer wtl;

        for ( long_t i = 0; i < rounds; ++i )
        {
            lnum = 0;
            auto in = net.get_random_sample();
            wtn.reset();

            for ( auto & l: layers )
            {
                wtl.reset();
                in = l->forward(std::move(in));
                double t = wtl.elapsed<double>();

                rout << "[layer_measurement] " << net_name
                     << " :: " << os << " :: " << lnum
                     << " :: " << t << std::endl;

                ++lnum;
            }

            double t = wtn.elapsed<double>();

            rout << "[network_measurement] " << net_name
                     << " :: " << os
                     << " :: " << t << std::endl;

            total += t;

            ++lnum;
        }

        total /= rounds;

    // rout << "## " << net_name << " :: starting benchmark for output size "
    //      << os << std::endl;

        rout << "[network_average] " << net_name
             << " :: " << os << " :: " << total << std::endl;

        double voxels = net.out_voxels();

        rout << "[network_throughput] " << net_name
             << " :: " << os << " :: " << (voxels/total) << std::endl;
    }

}

int main(int argc, char *argv[])
{
    net_name = std::string(argv[1]);

    std::string net_path    = "../networks/" + net_name + ".znni";
    std::string report_path = "../reports/" + net_name + ".host.report";

    std::ofstream ofs;
    ofs.open (report_path.c_str(), std::ofstream::out | std::ofstream::app);

    ofs << "--------------------------------------------\n\n";

    std::cout << net_path << "\n";

    long_t rounds = 4;
    if ( argc > 2 ) rounds = atoi(argv[2]);

    network_descriptor nd(net_path);

    for ( long_t i = 16; i < 1000; i += 16 )
    {
        os = vec3i(i,i,i);
        benchmark_network(nd, rounds, ofs);
    }
}
