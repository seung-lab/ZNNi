#include "cpu/pooling/naive.hpp"
#include "cpu/pooling/default.hpp"
#include "cpu/convolutional/direct.hpp"
#include "cpu/convolutional/naive.hpp"
#include "cpu/convolutional/padded_pruned_fft.hpp"
#include "gpu/convolutional/cudnn.hpp"
#include "gpu/convolutional/cufft.hpp"
#include "gpu/convolutional/ram/ram_cudnn.hpp"
#include "gpu/convolutional/padded_cufft.hpp"
#include "gpu/convolutional/padded_pruned_cufft.hpp"
#include "init.hpp"
#include <zi/time.hpp>

using namespace znn::fwd;

template<typename NetworkType>
void cpu_test_correctness( std::string const & name,
                           long_t n,
                           long_t fin,
                           long_t fout,
                           vec3i const & is,
                           vec3i const & ks,
                           real* km,
                           real* bs,
                           real* input,
                           real* expected_output)

{
    NetworkType net(n,fin,fout,is,ks,km,bs);

    host_array<real> in = get_array<real>(net.total_input_len);
    std::copy_n(input, net.total_input_len, in.get());

    zi::wall_timer wt;
    auto out = net.forward(std::move(in));

    std::cout << "CPU LAYER< " << name << " > "
              << wt.elapsed<double>() << "\t";


    real mdiff = 0;
    for ( long_t i = 0; i < net.total_output_len; ++i )
    {
        mdiff = std::max(mdiff,std::abs(expected_output[i]-out.get()[i]));
    }

    std::cout << ( mdiff ) << "\n";
}


template<typename NetworkType>
void gpu_test_correctness( std::string const & name,
                           long_t n,
                           long_t fin,
                           long_t fout,
                           vec3i const & is,
                           vec3i const & ks,
                           real* km,
                           real* bs,
                           real* input,
                           real* expected_output)

{
    NetworkType net(n,fin,fout,is,ks,km,bs);

    device_array<real> in_d = get_device_array<real>(net.total_input_len);

    zi::wall_timer wt;

    device_copy_n(input, net.total_input_len, in_d);
    auto out_d = net.forward(std::move(in_d));

    std::cout << "GPU LAYER< " << name << " > "
              << wt.elapsed<double>() << "\t";

    auto out = get_array<real>(net.total_output_len);

    checkCudaErrors( cudaMemcpy(out.get(), out_d.get(),
                                net.total_output_len * sizeof(float),
                                cudaMemcpyDeviceToHost) );

    real mdiff = 0;
    for ( long_t i = 0; i < net.total_output_len; ++i )
    {
        mdiff = std::max(mdiff,std::abs(expected_output[i]-out.get()[i]));
    }

    std::cout << ( mdiff ) << "\n";
}


void test_conv_layer()
{
    uniform_init ui(0.1);

    detail::random_number_generator_impl& rng =
        zi::singleton<detail::random_number_generator_impl>::instance();

    std::uniform_int_distribution<long_t> intdist(1,5);

    vec3i ws(intdist(rng.rng),intdist(rng.rng),intdist(rng.rng));
    std::uniform_int_distribution<long_t> intdist2(1,29);

    vec3i os(intdist2(rng.rng),intdist2(rng.rng),intdist2(rng.rng));

    std::uniform_int_distribution<long_t> intdist3(1,7);

    long_t n = intdist3(rng.rng);
    long_t l = intdist3(rng.rng);
    long_t l2 = intdist3(rng.rng);

    vec3i is = os + ws - vec3i::one;

    host_array<real> kernels = get_array<real>(l*l2*ws[0]*ws[1]*ws[2]);
    host_array<real> biases  = get_array<real>(l2);

    ui.initialize(kernels.get(), l*l2*ws[0]*ws[1]*ws[2]);
    ui.initialize(biases.get(), l2);

    cpu::naive_convolutional_layer  np(n,l,l2,is,ws, kernels.get(), biases.get());

    host_array<real> ina = get_array<real>(np.total_input_len);
    ui.initialize(ina.get(), np.total_input_len);

    host_array<real> inb = get_array<real>(np.total_input_len);
    std::copy_n(ina.get(), np.total_input_len, inb.get());

    auto r1 = np.forward(std::move(ina));

    cpu_test_correctness<cpu::direct_convolutional_layer>
        ( "DIRECT", n, l, l2, is, ws, kernels.get(), biases.get(),
          inb.get(), r1.get() );

    cpu_test_correctness<cpu::padded_pruned_fft_convolutional_layer>
        ( "PADDED PRUNED FFT", n, l, l2, is, ws, kernels.get(), biases.get(),
          inb.get(), r1.get() );

    cpu_test_correctness<gpu::ram_cudnn_convolutional_layer>
        ( "RAMGPU", n, l, l2, is, ws, kernels.get(), biases.get(),
          inb.get(), r1.get() );

    gpu_test_correctness<gpu::cudnn_convolutional_layer>
        ( " CUDNN", n, l, l2, is, ws, kernels.get(), biases.get(),
          inb.get(), r1.get() );

    gpu_test_correctness<gpu::cufft_convolutional_layer>
        ( " CUFFT", n, l, l2, is, ws, kernels.get(), biases.get(),
          inb.get(), r1.get() );

    gpu_test_correctness<gpu::padded_cufft_convolutional_layer>
        ( "PCUFFT", n, l, l2, is, ws, kernels.get(), biases.get(),
          inb.get(), r1.get() );

    gpu_test_correctness<gpu::padded_pruned_cufft_convolutional_layer>
        ( "PPCFFT", n, l, l2, is, ws, kernels.get(), biases.get(),
          inb.get(), r1.get() );


    // cpu::direct_convolutional_layer np(n,l,l2,is,ws, kernels.get(), biases.get());
    // //cpu::padded_pruned_fft_convolutional_layer pl(n,l,l2,is,ws, kernels.get(), biases.get());
    // gpu::ram_cudnn_convolutional_layer pl(n,l,l2,is,ws, kernels.get(), biases.get());
    // gpu::cudnn_convolutional_layer cudn(n,l,l2,is,ws, kernels.get(), biases.get());
    // gpu::cufft_convolutional_layer cudf(n,l,l2,is,ws, kernels.get(), biases.get());
    // gpu::padded_cufft_convolutional_layer cudfp(n,l,l2,is,ws, kernels.get(), biases.get());
    // gpu::padded_pruned_cufft_convolutional_layer cudfpp(n,l,l2,is,ws, kernels.get(), biases.get());


}

int main()
{
    while (1)
        test_conv_layer();

}
