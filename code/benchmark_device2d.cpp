#include "znn/util/network2d.hpp"
//#include "znn/util/network.hpp"
#include "znn/device/2d/conv.hpp"
#include "znn/device/2d/mfp.hpp"
#include "znn/device/2d/maxout.hpp"

#include <zi/time.hpp>

#include <string>
#include <fstream>
#include <sstream>

using namespace znn::fwd;

std::string net_name;
vec2i       os;
long_t      max_memory = static_cast<long_t>(11) * 1024 * 1024 * 1024; // GB
long_t      batch_size = 1;

inline void benchmark_network( network2d_descriptor & ndesc,
                               long_t rounds,
                               std::ofstream & rout )
{
    znni_network2d net(ndesc, batch_size, os);

    rout << "## " << net_name << " :: starting benchmark for output size "
         << batch_size << " x " << os << std::endl;

    std::vector<std::unique_ptr<device::twod::device_layer2d>> layers;

    long_t lnum = 0;

    long_t rm = 0;
    long_t wm = 0;

    try
    {
        for ( auto & l: net.layers() )
        {
            if ( l.descriptor.type == layer2d_type::convolutional )
            {
                layers.push_back(make_unique<device::twod::conv>
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
                layers.push_back(make_unique<device::twod::mfp>
                                 (l.batch_size,
                                  l.descriptor.num_inputs,
                                  l.in_size,
                                  l.descriptor.k_or_w_size));
            }
            else
            {
                layers.push_back(make_unique<device::twod::maxout>
                                 (l.batch_size,
                                  l.descriptor.num_inputs,
                                  l.descriptor.num_inputs
                                  /l.descriptor.num_outputs,
                                  l.in_size));
            }

            ++lnum;

            rm += layers.back()->resident_memory();
            wm = std::max(wm, layers.back()->working_memory());
        }
    }
    catch ( std::exception & e )
    {
        rout << "[network_exception] " << net_name
             << " :: " << batch_size << " x " << os << " :: "
             << " threw an exception: " << e.what() << std::endl;
        return;
    }

    rout << "[network_requirements] " << net_name
         << " :: " << batch_size << " x " << os << " :: "
         << "RM: " << rm/1024/1024
         << " MB WM: " << wm/1024/1024 << " MB" << std::endl;


    bool workable = true;
    for ( auto & l: layers )
    {
        if ( workable )
        {
            auto r = l->workable();
            workable = workable && r.first;
        }
    }

    if ( !workable )
    {
        rout << "[network_information] " << net_name
             << " :: " << batch_size << " x " << os << " :: IS NOT WORKABLE!" << std::endl;
        return;
    }

    if ( rm + wm < max_memory )
    {
        double total = 0;
        zi::wall_timer wtn;
        zi::wall_timer wtl;

        for ( long_t i = 0; i < rounds; ++i )
        {
            lnum = 0;
            auto inh = net.get_random_sample();

            wtn.reset();
	
	    //std::cout << "here\n";
	    //std::cout << "shape: " << net.in_shape() << "\n";

            device_tensor<float,4> in(net.in_shape());
            in = inh;
	
//            rout << "[2devi_measurement]: " << net_name 
//		 << " :: " << os << " :: " << wtl.elapsed<double>() << std::endl;

            for ( auto & l: layers )
            {
                wtl.reset();
                in = l->forward(std::move(in));
	        //in = std::move(zin);
                double t = wtl.elapsed<double>();

                rout << "[layer_measurement] " << net_name
                     << " :: " << batch_size << " x " << os << " :: " << lnum
                     << " :: " << t << std::endl;

                ++lnum;
            }

            host_tensor<float,4> result(in.shape()[0], in.shape()[1],
                                        in.shape()[2], in.shape()[3]);

            result = in;
            double t = wtn.elapsed<double>();

            rout << "[network_measurement] " << net_name
                     << " :: " << batch_size << " x " << os
                     << " :: " << t << std::endl;

            total += t;

            ++lnum;
        }

        total /= rounds;

        rout << "[network_average] " << net_name
             << " :: " << batch_size << " x " << os << " :: " << total << std::endl;

        double voxels = net.out_voxels();

        rout << "[network_throughput] " << net_name
             << " :: " << batch_size << " x " << os << " :: " << (voxels/total) << std::endl;
    }

}

void benchmark( std::string const & rep, long_t rounds )
{
    std::string net_path    = "../networks/" + net_name + ".znni";
    std::string report_path = "../reports/" + net_name + ".device." + rep + ".report";

    std::ofstream ofs;
    ofs.open (report_path.c_str(), std::ofstream::out | std::ofstream::app);

    ofs << "--------------------------------------------\n\n";

    std::cout << net_path << "\n";

    network2d_descriptor nd(net_path);

    for ( long_t i = 512; i < 11400; i += 512 )
    {
        os = vec2i(i,i);
        if ( os % nd.fragmentation() == vec2i::zero )
        {
	    for ( batch_size = 1; batch_size <= 4; batch_size *=2 )
            benchmark_network(nd, rounds, ofs);	
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
