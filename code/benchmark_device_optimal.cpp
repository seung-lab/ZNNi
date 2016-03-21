#include "znn/fusion/device/network.hpp"
#include "znn/fusion/network_data.hpp"
#include "znn/fusion/device/sub_network.hpp"
#include "znn/log.hpp"

#include <string>
#include <fstream>
#include <sstream>

// RUN THE FOLLOWING
// ./bin/benchmark_device_optimal m36 4
// ./bin/benchmark_device_optimal m56 4
// ./bin/benchmark_device_optimal m76 4
// ./bin/benchmark_device_optimal m96 4


using namespace znn::fwd;

std::string net_name;
vec3i       os;
long_t      max_memory = static_cast<long_t>(11) * 1024 * 1024 * 1024; // GB

inline void benchmark_network( network_descriptor & ndesc,
                               long_t rounds,
                               std::ofstream & rout )
{
    fusion::network_data nd(ndesc, 1, os);

    LOG(benchmark) << net_name << " :: starting os :: " << os;

    auto x = device::fusion::network::get(nd, true);

    if ( x )
    {
        try
        {
            double time = x->benchmark(rounds);
            double tput = os[2] * os[3] * os[4];
            tput /= time;

            rout << "[network_build] " << net_name
                 << " :: " << os << " :: " << x->name() << std::endl;

            rout << "[network_throughput] " << net_name
                 << " :: " << os << " :: " << (tput) << std::endl;
        }
        catch ( std::exception & e )
        {
            LOG(benchmark) << "benchmark filed: " << e.what();
        }
    }
    else
    {
        LOG(benchmark) << "not possible";
    }

}

void benchmark( std::string const & rep, long_t rounds )
{
    std::string net_path    = "../networks/" + net_name + ".znni";
    std::string report_path = "../reports/" + net_name + ".device." + rep + ".report";

    std::ofstream ofs;
    ofs.open (report_path.c_str(), std::ofstream::out | std::ofstream::app);

    ofs << "--------------------------------------------\n\n";
    logger.set_ostream(ofs);

    std::cout << net_path << "\n";

    network_descriptor nd(net_path);

    for ( long_t i = 4; i < 400; i += 4 )
    {
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
