#include "cpu/pooling/naive.hpp"
#include "cpu/pooling/default.hpp"
#include "init.hpp"
#include <zi/time.hpp>

using namespace znn::fwd;

void test_pooling_layer()
{
    uniform_init ui(4);

    detail::random_number_generator_impl& rng =
            zi::singleton<detail::random_number_generator_impl>::instance();

    std::uniform_int_distribution<long_t> intdist(1,4);

    vec3i ws(intdist(rng.rng),intdist(rng.rng),intdist(rng.rng));

    std::uniform_int_distribution<long_t> intdist2(1,20);

    vec3i os(intdist2(rng.rng),intdist2(rng.rng),intdist2(rng.rng));

    vec3i is = ws * os + ws - vec3i::one;

    long_t n = intdist2(rng.rng);
    long_t l = intdist2(rng.rng);


    task_package handle(1000000);

    cpu::naive_pooling_layer np(n,l,is,ws);
    cpu::pooling_layer       pl(handle,n,l,is,ws);

    host_array<real> ina = get_array<real>(np.total_input_len);
    ui.initialize(ina.get(), np.total_input_len);

    host_array<real> inb = get_array<real>(np.total_input_len);
    std::copy_n(ina.get(), np.total_input_len, inb.get());

    zi::wall_timer wt;

    auto r1 = np.forward(std::move(ina));

    std::cout << "Naive took: " << wt.elapsed<double>() << std::endl;

    wt.reset();
    auto r2 = pl.forward(std::move(inb));
    std::cout << "Default took: " << wt.elapsed<double>() << std::endl;

    real mdiff = 0;
    for ( long_t i = 0; i < np.total_output_len; ++i )
    {
        if (std::abs(r1.get()[i]-r2.get()[i]) > 0.0000001)
        {
            std::cout << ws << ' ' << n << ' ' << l << ' ' << os << " ---- ";
            std::cout << "     " << r1.get()[i] << ' ' << r2.get()[i] << "\n";
        }
        mdiff = std::max(mdiff, std::abs(r1.get()[i]-r2.get()[i]));
    }

    std::cout << ( ( mdiff <= 0.0000001 ) ?  "OK" : "ERROR") << "\n";

}

int main()
{
    while (1)
    test_pooling_layer();
    test_pooling_layer();
    test_pooling_layer();
}
