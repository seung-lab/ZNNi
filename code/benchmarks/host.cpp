#include "znn/host/v1/network.hpp"
#include "znn/util/network_data.hpp"

#include <zi/time.hpp>

#include <string>
#include <fstream>
#include <sstream>

// RUN THE FOLLOWING
// ./bin/benchmarks/host m36 3
// ./bin/benchmarks/host m56 3
// ./bin/benchmarks/host m76 3
// ./bin/benchmarks/host m96 3

using namespace znn::fwd;

std::string net_name  ;
vec3i       os        ;
long_t      max_memory;

template<typename Net, typename... Args>
inline std::unique_ptr<Net> get_network( network_descriptor const & desc,
                                         long_t batch,
                                         vec3i const & os,
                                         Args... args )
{
    network_data nd(desc, batch, os);
    return make_unique<Net>(nd, args...);
}

template<typename Net>
inline void benchmark_network( Net * net,
                               long_t rounds,
                               std::ofstream & rout )
{
    rout << "[benchmark] " << net_name << " :: starting os :: " << os
         << " memory required " << net->memory_required() << std::endl;

    return;
    if ( net->memory_required() < max_memory )
    {
        long_t total_outputs = 0;
        double time = 0;
        zi::wall_timer wt;
        for ( long_t i = 0; i < rounds; ++i )
        {
            host_tensor<float,5> in(rand_init,net->in_shape());
            wt.reset();

            in = net->forward(std::move(in));

            double t = wt.elapsed<double>();
            time += t;
            total_outputs += in.num_elements();

            rout << "[network_round] " << net_name
                 << " :: " << os << " :: " << t << std::endl;
        }

        double tput = time / total_outputs;

        rout << "[network_throughput] " << net_name
             << " :: " << os << " :: " << (tput)
             << " :: " << net->memory_required() << std::endl;
    }
    else
    {
        rout << "[benchmark] " << net_name << " :: starting os :: " << os
             << " memory required " << net->memory_required()
             << " TOO BIG!" << std::endl;
    }
}


template<typename Net, typename... Args>
inline long_t max_size( network_descriptor const & desc, long_t batch,
                        long_t low, long_t high, long_t factor,
                        long_t max_memory, Args... args )
{
    long_t mid = ( low + high ) / 2;

    if ( low == mid )
    {
        return low;
    }

    vec3i os(mid*factor,mid*factor,mid*factor);

    auto net = get_network<Net>(desc, batch, os, args...);

    if ( net->memory_required() > max_memory )
    {
        return max_size<Net>(desc,batch,low,mid,factor,max_memory,args...);
    }
    else
    {
        return max_size<Net>(desc,batch,mid,high,factor,max_memory,args...);
    }
}

std::vector<long_t> get_sizes( long_t max_size, long_t factor )
{
}

int main(int argc, char *argv[])
{
    net_name = std::string(argv[1]);

    long_t rounds = 4;
    if ( argc > 2 ) rounds = atoi(argv[2]);

    max_memory = static_cast<long_t>(240) * 1024 * 1024 * 1024; // GB
    if ( argc > 3 ) max_memory = static_cast<long_t>(atoi(argv[3])) << 30;


    std::string net_path    = "../networks/" + net_name + ".znni";
    std::string report_path = "../reports/" + net_name + ".benchmark.host";

    network_descriptor nd(net_path);

    // std::cout << "MAX SIZE: " << max_size<host::v1::network>
    //     (nd, 1, 1, 1000, nd.fragmentation()[0], max_memory);

    std::ofstream ofs;
    ofs.open (report_path.c_str(), std::ofstream::out | std::ofstream::app);

    ofs << "--------------------------------------------\n\n";

    // std::cout << net_path << "\n";


    for ( long_t i = 16; i < 700; i += 16 )
    {
        os = vec3i(i,i,i);
        auto net = get_network<host::v1::network>(nd,1,os);
        benchmark_network(net.get(), rounds, ofs);
    }
}
