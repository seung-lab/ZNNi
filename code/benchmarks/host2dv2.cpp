#include "znn/host/common/thread_pin.hpp"
#include "znn/util/network2d.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/2dv2/mfp_serial.hpp"
#include "znn/host/2dv2/maxout_serial.hpp"
#include "znn/host/2dv2/fft_conv_serial.hpp"

#include <zi/time.hpp>

#include <string>
#include <fstream>
#include <sstream>

using namespace znn::fwd;

std::string net_name;
vec2i       os;
long_t      max_memory = static_cast<long_t>(240) * 1024 * 1024 * 1024; // GB
long_t      batch_size = 1;
std::mutex  mtx;

inline void benchmark_network( network2d_descriptor & ndesc,
                               long_t rounds,
                               std::ofstream & rout )
{
    znni_network2d net(ndesc, 1, os);

    rout << "## " << net_name << " :: starting benchmark for output size "
         << os << ' ' << " BATCH: " << batch_size << std::endl;

    std::vector<std::unique_ptr<host::twod::host_layer2d>> layers;

    long_t lnum = 0;

    long_t rm = 0;
    long_t wm = 0;

    long_t workspace_size = 0;
    long_t inout_size = 0;

    try
    {
        for ( auto & l: net.layers() )
        {
            if ( l.descriptor.type == layer2d_type::convolutional )
            {
                layers.push_back(make_unique<host::twod::fft_conv2d_serial>
                              (l.batch_size,
                               l.descriptor.num_inputs,
                               l.descriptor.num_outputs,
                               l.in_size,
                               l.descriptor.k_or_w_size,
                               l.random_kernels().data(),
                               l.random_biases().data()));

            }
            else if ( l.descriptor.type == layer2d_type::pooling )
            {
                layers.push_back(make_unique<host::twod::mfp2d_serial>
                                 (l.batch_size,
                                  l.descriptor.num_inputs,
                                  l.in_size,
                                  l.descriptor.k_or_w_size));
            }
            else
            {
                layers.push_back(make_unique<host::twod::maxout2d_serial>
                                 (l.batch_size,
                                  l.descriptor.num_inputs,
                                  l.descriptor.num_inputs
                                  /l.descriptor.num_outputs,
                                  l.in_size));
            }

            ++lnum;

            rm += layers.back()->resident_memory();
            wm = std::max(wm, layers.back()->working_memory());

            inout_size = std::max(inout_size, layers.back()->input_memory);
            inout_size = std::max(inout_size, layers.back()->output_memory);
            workspace_size = std::max(workspace_size,
                                      layers.back()->workspace_size());
        }
    }
    catch ( std::exception & e )
    {
        rout << "[network_exception] " << net_name
             << " :: " << os << " :: "
             << " threw an exception: " << e.what() << std::endl;
        return;
    }

    long_t nthreads = host::architectire::available_threads();

    long_t total_mem = nthreads * (2 * inout_size + workspace_size);

    rout << "[network_requirements] " << net_name
         << " :: " << os << " :: "
         << "RM: " << rm/1024/1024
         << " MB WM: " << total_mem/1024/1024 << " MB" << std::endl;

    if ( rm + total_mem < max_memory )
    {
        host_array<float> inout[nthreads][2];
        host_array<float> wspace[nthreads];
        host_tensor<float,4> results[nthreads];

        for ( long_t t = 0; t < nthreads; ++t )
        {
            inout[t][0] = host_array<float>(rand_init,inout_size/4);
            inout[t][1] = host_array<float>(rand_init,inout_size/4);
            wspace[t]   = host_array<float>(rand_init,workspace_size/4);
            results[t]
                = host_tensor<float,4>(rand_init,layers.back()->output_shape);
        }

        auto fn = [&](float* in, float* out, long_t t)
            {
                size_t lnum = 0;
                for ( auto & l: layers )
                {
                    if ( lnum == 0 )
                    {
                        l->forward(in,
                                   inout[t][(lnum+1)%2].data(),
                                   wspace[t].data());
                    }
                    else if ( lnum == layers.size() - 1 )
                    {
                        l->forward(inout[t][lnum%2].data(),
                                   out,
                                   wspace[t].data());
                    }
                    else
                    {
                        l->forward(inout[t][lnum%2].data(),
                                   inout[t][(lnum+1)%2].data(),
                                   wspace[t].data());
                    }

                    ++lnum;
                }
            };

        double total = 0;
        zi::wall_timer wtn;

        for ( long_t i = 0; i < rounds; ++i )
        {
            std::vector<host_tensor<float,4>> ins(nthreads);
            for ( long_t t = 0; t < nthreads; ++t )
            {
                ins[t] = net.get_random_sample();
            }

            wtn.reset();

            tbb::parallel_for( static_cast<long_t>(0), nthreads,
                               [&](long_t i)
                               {
                                   fn(ins[i].data(), results[i].data(),i);
                               });

            double t = wtn.elapsed<double>();

            rout << "[network_measurement] " << net_name
                     << " :: " << os
                     << " :: " << t << std::endl;

            total += t;
        }

        total /= rounds;

        rout << "[network_average] " << net_name
             << " :: " << os << " :: " << total << std::endl;

        double voxels = net.out_voxels();

        rout << "[network_throughput] " << net_name
             << " :: " << os << " :: " << (nthreads*voxels/total) << std::endl;

    }
}

void benchmark( std::string const & rep, long_t rounds )
{
    std::string net_path    = "../networks/" + net_name + ".znni";
    std::string report_path = "../reports/" + net_name + ".host." + rep + ".report";

    std::ofstream ofs;
    ofs.open (report_path.c_str(), std::ofstream::out | std::ofstream::app);

    ofs << "--------------------------------------------\n\n";

    std::cout << net_path << "\n";

    network2d_descriptor nd(net_path);

    for ( long_t i = 256; i < 1140; i *= 2 )
    {
        os = vec2i(i-48,i-48);
        if ( os % nd.fragmentation() == vec2i::zero )
        {
            for ( batch_size = 1; batch_size <= 32; batch_size *= 2 )
            {
                for ( batch_size = 1; batch_size <= 64; batch_size *=2 )
                    benchmark_network(nd, rounds, ofs);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    net_name = std::string(argv[1]);

    long_t rounds = 4;
    if ( argc > 2 ) rounds = atoi(argv[2]);

    benchmark("2d", rounds);
}
