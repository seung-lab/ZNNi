#include "cpu/pooling/naive.hpp"
#include "cpu/pooling/default.hpp"
#include "cpu/convolutional/direct.hpp"
#include "cpu/convolutional/naive.hpp"
#include "cpu/convolutional/padded_pruned_fft.hpp"
#include "init.hpp"
#include <zi/time.hpp>

using namespace znn::fwd;

// Usable AlmostEqual function
bool nearly_equal(float A, float B, int32_t maxUlps)
{
    // Make sure maxUlps is non-negative and small enough that the
    // default NAN won't compare as equal to anything.
    assert(maxUlps > 0 && maxUlps < 4 * 1024 * 1024);
    int32_t aInt = *(int32_t*)&A;
    // Make aInt lexicographically ordered as a twos-complement int
    if (aInt < 0)
        aInt = 0x80000000 - aInt;
    // Make bInt lexicographically ordered as a twos-complement int
    int32_t bInt = *(int32_t*)&B;
    if (bInt < 0)
        bInt = 0x80000000 - bInt;
    int32_t intDiff = abs(aInt - bInt);
    if (intDiff <= maxUlps)
        return true;
    return false;
}

void test_conv_layer()
{
    uniform_init ui(0.1);

    detail::random_number_generator_impl& rng =
            zi::singleton<detail::random_number_generator_impl>::instance();

    std::uniform_int_distribution<long_t> intdist(1,5);

    vec3i ws(intdist(rng.rng),intdist(rng.rng),intdist(rng.rng));

    std::uniform_int_distribution<long_t> intdist2(1,40);

    vec3i os(intdist2(rng.rng),intdist2(rng.rng),intdist2(rng.rng));

    vec3i is = os + ws - vec3i::one;

    long_t n = intdist2(rng.rng);
    long_t l = intdist2(rng.rng);
    long_t l2 = intdist2(rng.rng);


    task_package handle(1000000);

    host_array<real> kernels = get_array<real>(l*l2*ws[0]*ws[1]*ws[2]);
    host_array<real> biases  = get_array<real>(l2);

    ui.initialize(kernels.get(), l*l2*ws[0]*ws[1]*ws[2]);
    ui.initialize(biases.get(), l2);

    //cpu::naive_convolutional_layer  np(n,l,l2,is,ws, kernels.get(), biases.get());
    cpu::direct_convolutional_layer np(handle,n,l,l2,is,ws, kernels.get(), biases.get());

    cpu::padded_pruned_fft_convolutional_layer pl(handle,n,l,l2,is,ws, kernels.get(), biases.get());

    host_array<real> ina = get_array<real>(np.total_input_len);
    ui.initialize(ina.get(), np.total_input_len);

    host_array<real> inb = get_array<real>(np.total_input_len);
    std::copy_n(ina.get(), np.total_input_len, inb.get());

    std::cout << "   NETL " << ws << ' ' << n << ' '
              << l << ' ' << l2 << ' ' << os << "\n";

    zi::wall_timer wt;
    auto r1 = np.forward(std::move(ina));
    std::cout << "    Naive took: " << wt.elapsed<double>() << std::endl;

    wt.reset();
    auto r2 = pl.forward(std::move(inb));
    std::cout << "    Default took: " << wt.elapsed<double>() << std::endl;

    real mdiff = 0;

    bool ok = true;

    for ( long_t i = 0; i < np.total_output_len; ++i )
    {
        mdiff = std::max(mdiff,std::abs(r1.get()[i]- r2.get()[i]));

        if ( !nearly_equal(r1.get()[i], r2.get()[i], 1000000) )
        {
            //std::cout << "     " << r1.get()[i] << ' ' << r2.get()[i] << "\n";
        }

        ok &=  nearly_equal(r1.get()[i], r2.get()[i], 1000000);
    }

    std::cout << ( mdiff ) << "\n";

}

int main()
{
    while (1)
        test_conv_layer();
}
