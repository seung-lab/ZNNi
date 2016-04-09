#include "znn/device/fusion/network.hpp"
#include "znn/util/network_data.hpp"

#include <string>
#include <fstream>
#include <sstream>

// RUN THE FOLLOWING
// ./bin/benchmark_fusion_b m36 4 6
// ./bin/benchmark_fusion_b m56 4 6


using namespace znn::fwd;

std::string net_name;
vec3i       os;
long_t      max_memory = static_cast<long_t>(240) * 1024 * 1024 * 1024; // GB

std::ofstream ofs;

zi::wall_timer wt;
double total_time;
long_t round_num ;

long_t cuttoff;
long_t start;

void callback( host_tensor<float,5> )
{
    double t = wt.lap<double>();
    if ( round_num > 0 )
    {
        ofs << "[network_round] " << net_name
            << " :: " << os << " :: " << t << std::endl;
        total_time += t;
    }
    ++round_num;
}

inline void benchmark_network( network_descriptor & ndesc,
                               long_t rounds,
                               std::ofstream & rout )
{
    network_data nd(ndesc, 1, os);

    wt.reset();
    total_time = 0;
    round_num  = 0;

    bool didit = true;

    {
        device::fusion::network<device::fusion::gemm_it> net(nd,cuttoff,callback);

        LOG(benchmark) << net_name << " :: starting os :: " << os
                       << " memory required " << net.memory_required();

        if ( net.memory_required() < max_memory )
        {
            for ( long_t i = 0; i < rounds; ++i )
            {
                host_tensor<float,5> in(rand_init,net.in_shape());
                net.forward(std::move(in));
            }
        }
        else
        {
            LOG(benchmark) << "too big";
            didit = false;
        }
    }

    if ( didit )
    {
        total_time /= (rounds-1);
        double tput = os[0] * os[1] * os[2];
        tput /= total_time;
        rout << "[network_throughput] " << net_name
             << " :: " << os << " :: " << (tput) << std::endl;
    }
}

void benchmark( std::string const & rep, long_t rounds )
{
    std::string net_path    = "../networks/" + net_name + ".znni";
    std::string report_path = "../reports/" + net_name + ".fusion_b." + rep + ".report";

    ofs.open (report_path.c_str(), std::ofstream::out | std::ofstream::app);

    ofs << "--------------------------------------------\n\n";
    logger.set_ostream(ofs);

    std::cout << net_path << "\n";

    network_descriptor nd(net_path);

    for ( long_t i = start; i < 1000; i += 16 )
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

    cuttoff = 4;
    if ( argc > 3 ) cuttoff = atoi(argv[3]);

    start = 16;
    if ( argc > 4 ) start = atoi(argv[4]);

    benchmark("optimal", rounds);
}
