#include "utils/network_descriptor.hpp"
#include "cpu/layers.hpp"
#include "cpu/convolutional/padded_pruned_fft_auto.hpp"
#include "gpu/convolutional/ram/ram_cudnn.hpp"
#include "gpu/layers.hpp"
#include "cpu/layers.hpp"
#include "tbb/layers.hpp"

#include <zi/time.hpp>
#include <condition_variable>

using namespace znn::fwd;

struct cpu_sample
{
    host_array<real> prepare( host_array<real> s, long_t )
    {
        return s;
    }

    host_array<real> fetch( host_array<real> s, long_t )
    {
        return s;
    }
};

struct gpu_sample
{
    device_array<float> prepare( host_array<float> s, long_t l )
    {
        device_array<float> r = get_device_array<float>(l);
        checkCudaErrors( cudaMemcpy(r.get(), s.get(), l*sizeof(float),
                                    cudaMemcpyHostToDevice) );
        return r;
    }

    host_array<real> fetch( device_array<real> s, long_t l )
    {
        host_array<float> r = get_array<float>(l);
        checkCudaErrors( cudaMemcpy(r.get(), s.get(), l*sizeof(float),
                                    cudaMemcpyDeviceToHost) );
        return r;
    }
};

template<class Sampler, class Conv, class Pool>
struct benchmark
{
    typedef Sampler sampler_t;
    typedef Conv    conv_t   ;
    typedef Pool    pool_t   ;

    typedef typename conv_t::layer_type  layer_t;
    typedef typename conv_t::array_type  array_t;

    double operator()( znni_network & net, long_t rounds = 2 ) const
    {
        sampler_t sampler;

        std::vector<std::unique_ptr<layer_t>> layers;

        for ( auto const & l: net.layers() )
        {
            if ( l.descriptor.type == layer_type::convolutional )
            {
                layers.push_back(std::unique_ptr<layer_t>
                                 (new conv_t
                                  (l.batch_size,
                                   l.descriptor.num_inputs,
                                   l.descriptor.num_outputs,
                                   l.in_size,
                                   l.descriptor.k_or_w_size,
                                   l.kernels.get(),
                                   l.biases.get())));
            }
            else
            {
                layers.push_back(std::unique_ptr<layer_t>
                                 (new pool_t
                                  (l.batch_size,
                                   l.descriptor.num_outputs,
                                   l.in_size,
                                   l.descriptor.k_or_w_size)));
            }
        }

        double total_time = 0;

        for ( long_t r = 0; r < rounds; ++r )
        {
            zi::wall_timer wta, wt;
            auto x = net.get_random_sample();

            wta.reset();

            wt.reset();
            auto y = sampler.prepare(std::move(x), net.in_len());
            std::cout << "Sample copy took\t" << wt.elapsed<double>()
                      << std::endl;


            for ( size_t i = 0; i < layers.size(); ++i )
            {
                wt.reset();
                y = layers[i]->forward(std::move(y));
                std::cout << "Layer " << (i+1) << " took\t" << wt.elapsed<double>()
                          << std::endl;
            }


            wt.reset();
            x = sampler.fetch(std::move(y), net.out_len());

            total_time += wta.elapsed<double>();

            std::cout << "Result copy took\t" << wt.elapsed<double>()
                      << std::endl;

            std::cout << "Total: " << wta.elapsed<double>()
                      << std::endl << std::endl;
        }

        return total_time / rounds;

    }
};


template<class CPUConv, class CPUPool, class GPUConv, class GPUPool>
struct benchmark_fusion
{
    typedef cpu_sample sampler_t ;
    typedef CPUConv    cpu_conv_t;
    typedef CPUPool    cpu_pool_t;

    typedef GPUConv    gpu_conv_t;
    typedef GPUPool    gpu_pool_t;

    typedef typename cpu_conv_t::layer_type  cpu_layer_t;
    typedef typename cpu_conv_t::array_type  cpu_array_t;

    typedef typename gpu_conv_t::layer_type  gpu_layer_t;
    typedef typename gpu_conv_t::array_type  gpu_array_t;

    double operator()( znni_network & net,
                       long_t cuttoff,
                       long_t rounds = 2 ) const
    {
        sampler_t sampler;

        std::vector<std::unique_ptr<cpu_layer_t>> cpu_layers;
        std::vector<std::unique_ptr<gpu_layer_t>> gpu_layers;

        long_t gpu_batch_size = 0;

        long_t curr_layer = 0;
        for ( auto const & l: net.layers() )
        {
            if ( curr_layer < cuttoff )
            {
                if ( l.descriptor.type == layer_type::convolutional )
                {
                    cpu_layers.push_back(std::unique_ptr<cpu_layer_t>
                                         (new cpu_conv_t
                                          (l.batch_size,
                                           l.descriptor.num_inputs,
                                           l.descriptor.num_outputs,
                                           l.in_size,
                                           l.descriptor.k_or_w_size,
                                           l.kernels.get(),
                                           l.biases.get())));
                }
                else
                {
                    cpu_layers.push_back(std::unique_ptr<cpu_layer_t>
                                         (new cpu_pool_t
                                          (l.batch_size,
                                           l.descriptor.num_outputs,
                                           l.in_size,
                                           l.descriptor.k_or_w_size)));
                }
            }
            else
            {
                if ( cuttoff == curr_layer )
                {
                    gpu_batch_size = l.batch_size;
                }
                if ( l.descriptor.type == layer_type::convolutional )
                {
                    gpu_layers.push_back(std::unique_ptr<gpu_layer_t>
                                         (new gpu_conv_t
                                          (l.batch_size/gpu_batch_size,
                                           l.descriptor.num_inputs,
                                           l.descriptor.num_outputs,
                                           l.in_size,
                                           l.descriptor.k_or_w_size,
                                           l.kernels.get(),
                                           l.biases.get())));
                }
                else
                {
                    gpu_layers.push_back(std::unique_ptr<gpu_layer_t>
                                         (new gpu_pool_t
                                          (l.batch_size/gpu_batch_size,
                                           l.descriptor.num_outputs,
                                           l.in_size,
                                           l.descriptor.k_or_w_size)));
                }
            }
            ++ curr_layer;
        }

        std::mutex mtx;
        std::condition_variable cpu_cv;
        std::condition_variable gpu_cv;

        long_t cpu_done = 0;
        long_t gpu_done = 0;

        host_array<real> handover[2];

        std::thread gpu_thread( [&]() {

                double tot_time = 0;

                zi::wall_timer wt;

                long_t tot_in_len  = gpu_layers.front()->total_input_len;
                long_t tot_out_len = gpu_layers.back()->total_output_len;

                for ( long_t r = 0; r < rounds; ++r )
                {
                    {
                        std::unique_lock<std::mutex> g(mtx);
                        while ( cpu_done <= r )
                        {
                            gpu_cv.wait(g);
                        }
                    }

                    auto input  = std::move(handover[r%2]);
                    auto output = get_array<real>
                        (tot_out_len*gpu_batch_size);

                    for ( long_t i = 0; i < gpu_batch_size )
                    {
                        device_array<float> r = get_device_array<float>(tot_in_len);
                        checkCudaErrors( cudaMemcpy(r.get(), input.get()
                                                    + i * tot_in_len,
                                                    tot_in_len*sizeof(float),
                                                    cudaMemcpyHostToDevice) );

                        for ( auto & l: gpu_layers )
                        {
                            r = l->forward(std::move(r));
                        }

                        checkCudaErrors( cudaMemcpy(output.get()
                                                    + i * tot_out_len,
                                                    r.get(),
                                                    tot_out_len*sizeof(float),
                                                    cudaMemcpyDeviceToHost) );
                    }

                    {
                        std::unique_lock<std::mutex> g(mtx);
                        ++gpu_done;
                        cpu_cv.notify_all();
                    }

                    double tt = wt.lap<double>();

                    if ( r > 0 )
                    {
                        tot_time += tt;
                        tt /= net.get_total_out_len();
                        std::cout << "AS: " << net.get_out_size()
                                  << ' ' << tt << std::endl;
                    }
                }

                tot_time /= net.get_total_out_len();
                tot_time /= (rounds-1);

                std::cout << "OS: " << net.get_out_size()
                          << ' ' << tot_time << std::endl;

            });

        for ( long_t r = 0; r < rounds; ++r )
        {
            {
                std::unique_lock<std::mutex> g(mtx);
                while ( gpu_done + 2 <= r )
                {
                    cpu_cv.wait(g);
                }
            }

            auto x = net.get_random_sample();

            for ( auto & l: cpu_layers )
            {
                x = l->forward(std::move(l));
            }

            handover[r%2] = std::move(x);

            {
                std::unique_lock<std::mutex> g(mtx);
                ++cpu_done;
                gpu_cv.notify_all();
            }
        }

        gpu_thread.join();
    }
};


int main(int argc, char *argv[])
{

    std::string f(argv[1]);

    vec3i os;

    if ( argc > 2 ) os[0] = atoi(argv[2]);
    if ( argc > 3 ) os[1] = atoi(argv[3]);
    if ( argc > 4 ) os[2] = atoi(argv[4]);

    long_t cutoff = 3;
    if ( argc > 2 ) cutoff = atoi(argv[2]);

    long_t rounds = 5;
    if ( argc > 3 ) rounds = atoi(argv[3]);

    network_descriptor nd(f);

    for ( long_t x = 16; x < 400; x += 16 )
    {
        os[0] = x; os[1] = x; os[2] = x;
        znni_network net(nd, bs, os);

        benchmark_fusion<
            znn::fwd::tbb::padded_pruned_fft_auto_convolutional_layer,
            znn::fwd::tbb::pooling_layer,
            gpu::padded_pruned_cufft_convolutional_layer,
            gpu::cudnn_pooling_layer> b;

        b(net,cuttoff,rounds);
    }

}
