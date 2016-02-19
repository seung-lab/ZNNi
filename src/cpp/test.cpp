#include "utils/network_descriptor.hpp"
#include "cpu/layers.hpp"
#include "cpu/convolutional/padded_pruned_fft_auto.hpp"
#include "gpu/convolutional/ram/ram_cudnn.hpp"
#include "gpu/layers.hpp"

#include <zi/time.hpp>

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

    void operator()( znni_network & net ) const
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


        for ( long_t r = 0; r < 100; ++r )
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
            std::cout << "Result copy took\t" << wt.elapsed<double>()
                      << std::endl;

            std::cout << "Total: " << wta.elapsed<double>()
                      << std::endl << std::endl;
        }

    }
};


int main(int argc, char *argv[])
{
    //auto x = znn::fwd::layer_type::convolutional;

    std::string f(argv[1]);

    vec3i os(32,32,32);

    if ( argc > 2 ) os[0] = atoi(argv[2]);
    if ( argc > 3 ) os[1] = atoi(argv[3]);
    if ( argc > 4 ) os[2] = atoi(argv[4]);

    network_descriptor nd(f);
    znni_network       net(nd, 1, os);

    {
        benchmark<cpu_sample,
                  gpu::ram_cudnn_convolutional_layer,
                  cpu::pooling_layer> b;

        b(net);
    }

    {
        benchmark<cpu_sample,
                  cpu::padded_pruned_fft_convolutional_layer,
                  cpu::pooling_layer> b;

        b(net);
    }

}
