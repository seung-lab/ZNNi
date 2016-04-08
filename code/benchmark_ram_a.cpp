#include "znn/fusion/device/network.hpp"
#include "znn/fusion/network_data.hpp"
#include "znn/fusion/device/sub_network.hpp"
#include "znn/log.hpp"
#include "znn/device/tail/sub_network.hpp"
#include "znn/device/tail/network.hpp"
#include "znn/device/ram/network.hpp"
#include "znn/util/network_data.hpp"

#include <string>
#include <fstream>
#include <sstream>

// RUN THE FOLLOWING
// ./bin/benchmark_ram_a m76 4
// ./bin/benchmark_ram_a m96 4


using namespace znn::fwd;

std::string net_name;
vec3i       os;
long_t      max_memory = static_cast<long_t>(240) * 1024 * 1024 * 1024; // GB

inline void benchmark_network( network_descriptor & ndesc,
                               long_t rounds,
                               std::ofstream & rout )
{
    network_data nd(ndesc, 1, os);
    device::ram::network<device::ram::fft_it> net(nd,4);

    LOG(benchmark) << net_name << " :: starting os :: " << os
                   << " memory required " << net.memory_required();

    if ( net.memory_required() < max_memory )
    {
        double time = 0;
        zi::wall_timer wt;
        for ( long_t i = 0; i < rounds; ++i )
        {
            host_tensor<float,5> in(rand_init,net.in_shape());
            wt.reset();
            in = net.forward(std::move(in));
            double t = wt.elapsed<double>();
            time += t;
            rout << "[network_round] " << net_name
                 << " :: " << os << " :: " << t << std::endl;
        }

        time /= rounds;

        double tput = os[0] * os[1] * os[2];
        tput /= time;


        rout << "[network_throughput] " << net_name
             << " :: " << os << " :: " << (tput) << std::endl;
    }
    else
    {
        LOG(benchmark) << "too big";
    }
}

void benchmark( std::string const & rep, long_t rounds )
{
    std::string net_path    = "../networks/" + net_name + ".znni";
    std::string report_path = "../reports/" + net_name + ".ram_a." + rep + ".report";

    std::ofstream ofs;
    ofs.open (report_path.c_str(), std::ofstream::out | std::ofstream::app);

    ofs << "--------------------------------------------\n\n";
    logger.set_ostream(ofs);

    std::cout << net_path << "\n";

    network_descriptor nd(net_path);

    for ( long_t i = 16; i < 1000; i += 16 )
    {
        if ( i > 200 ) i += 16;
        os = vec3i(i,i,i);
        if ( os % nd.fragmentation() == vec3i::zero )
        {
            benchmark_network(nd, rounds, ofs);
        }
    }

    logger.set_ostream(std::cout);
}

int main(int argc, char *argv[])
{
    net_name = std::string(argv[1]);

    long_t rounds = 4;
    if ( argc > 2 ) rounds = atoi(argv[2]);

    benchmark("optimal", rounds);
}
