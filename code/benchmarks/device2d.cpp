#include "znn/util/network2d.hpp"
#include "znn/device/2d/conv.hpp"
#include "znn/device/2d/mfp.hpp"
#include "znn/device/2d/maxout.hpp"

#include <zi/time.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <thread>
#include <condition_variable>

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
        // get all the samples
        std::vector<host_tensor<float,4>> inputs(rounds);

        for ( long_t i = 0; i < rounds; ++i )
        {
            inputs[i] = net.get_random_sample();
        }

        std::mutex mtx;
        std::condition_variable feed_cv;
        std::condition_variable exec_cv;
        std::condition_variable fetch_cv;

        bool has_feed   = false;
        bool has_result = false;

        device_tensor<float,4> feed;
        device_tensor<float,4> result;

        double total_voxels = 0;
        zi::wall_timer wtn;

        auto feeder_thread_impl = [&]() {
            wtn.reset();
            for ( long_t i = 0; i < rounds; ++i )
            {
                device_tensor<float,4> to_device(net.in_shape());
                to_device = inputs[i];

                {
                    std::unique_lock<std::mutex> g(mtx);
                    while ( has_feed )
                    {
                        feed_cv.wait(g);
                    }

                    feed = std::move(to_device);
                    has_feed = true;
                    exec_cv.notify_one();
                }
            }
        };

        auto fetch_thread_impl = [&]() {

            device_tensor<float,4> to_host;

            for ( long_t i = 0; i < rounds; ++i )
            {
                {
                    std::unique_lock<std::mutex> g(mtx);
                    while ( !has_result )
                    {
                        fetch_cv.wait(g);
                    }

                    to_host = std::move(result);
                    has_result = false;
                    exec_cv.notify_one();
                }

                inputs[i] = host_tensor<float,4>(to_host.shape_vec());
                inputs[i] = to_host;
                to_host.reset();

                total_voxels += net.out_voxels();
                double total_time = wtn.elapsed<double>();
                double num_rounds = (i+1);

                rout << "[network_time] " << net_name
                << " :: " << batch_size << " x " << os << " :: "
                << i << " :: " << (num_rounds/total_time) << std::endl;

                rout << "[network_throughput] " << net_name
                << " :: " << batch_size << " x " << os << " :: "
                << i << " :: " << (total_voxels/total_time) << std::endl;
            }
        };

        // start the threads;
        std::thread t1(feeder_thread_impl);
        std::thread t2(fetch_thread_impl);

        for ( long_t i = 0; i < rounds; ++i )
        {
            device_tensor<float,4> in;

            {
                std::unique_lock<std::mutex> g(mtx);
                while ( !has_feed )
                {
                    exec_cv.wait(g);
                }

                in = std::move(feed);
                has_feed = false;
                feed_cv.notify_one();
            }

            for ( auto & l: layers )
            {
                in = l->forward(std::move(in));
            }

            {
                std::unique_lock<std::mutex> g(mtx);
                while ( has_result )
                {
                    exec_cv.wait(g);
                }

                result = std::move(in);
                has_result = true;
                fetch_cv.notify_one();
            }
        }

        t1.join();
        t2.join();
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

    for ( long_t i = 256; i < 11400; i *= 2 )
    {
        os = vec2i(i-48,i-48);
        if ( os % nd.fragmentation() == vec2i::zero )
        {
            for ( batch_size = 1; batch_size <= 64; batch_size *=2 )
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
