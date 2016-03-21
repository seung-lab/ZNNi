#include "znn/device/v2/best_device_network.hpp"

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
// ./benchmark_best_device m36 8
// ./benchmark_best_device m56 8
// ./benchmark_best_device m76 8
// ./benchmark_best_device m96 8

using namespace znn::fwd;

std::string net_name;
vec3i       os;
long_t      max_memory = static_cast<long_t>(240) * 1024 * 1024 * 1024; // GB

inline void benchmark_network( network_descriptor & ndesc,
                               long_t rounds,
                               std::ofstream & rout )
{

    rout << "## " << net_name << " :: starting benchmark for output size "
         << os << std::endl;

    try
    {
        device::v2::network_data nd(ndesc, 1, os);
        device::v2::best_device_network net(nd);

        double time = 0;

        for ( long_t i = 0; i < rounds; ++i )
        {
            std::cout << "BENCH:" << std::endl;
            double t = net.benchmark();
            rout << "[network_measurement] " << net_name
                 << " :: " << os
                 << " :: " << t << std::endl;
            time += t;
        }

        time /= rounds;

        rout << "[network_average] " << net_name
             << " :: " << os << " :: " << time << std::endl;

        double voxels = os[0] * os[1] * os[2];

        rout << "[network_throughput] " << net_name
             << " :: " << os << " :: " << (voxels/time) << std::endl;
    }
    catch (...)
    {
        rout << "## " << net_name << " :: " << os
             << " :: EXCEPTION!" << std::endl;
    }

}

int main(int argc, char *argv[])
{
    net_name = std::string(argv[1]);

    std::string net_path    = "../networks/" + net_name + ".znni";
    std::string report_path = "../reports/" + net_name + ".best_device.report";

    std::ofstream ofs;
    ofs.open (report_path.c_str(), std::ofstream::out | std::ofstream::app);

    ofs << "--------------------------------------------\n\n";

    std::cout << net_path << "\n";

    long_t rounds = 4;
    if ( argc > 2 ) rounds = atoi(argv[2]);

    network_descriptor nd(net_path);

    for ( long_t i = 4; i < 1000; i += 4 )
    {
        os = vec3i(i,i,i);
        benchmark_network(nd, rounds, ofs);
    }
}
