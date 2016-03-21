#include "znn/host/v1/naive_boost_mfp.hpp"
#include "znn/host/v1/naive_conv.hpp"
#include "znn/host/v1/naive_boost_pool.hpp"
#include "znn/device/v1/fft_conv.hpp"
#include "znn/device/v1/cudnn_conv.hpp"
#include "znn/device/v1/cudnn_pool.hpp"
#include "znn/device/v1/cudnn_mfp.hpp"
#include "znn/device/v1/cudnn_no_precomp_gemm_conv.hpp"

using namespace znn::fwd;

template<class T1, class T2>
inline float max_abs_diff( T1 & t1, T2 & t2 )
{
    float* d1 = reinterpret_cast<float*>(
        detail::tensor::malloc(t1.num_elements() * sizeof(float),
                               from_host));
    float* d2 = reinterpret_cast<float*>(
        detail::tensor::malloc(t2.num_elements() * sizeof(float),
                               from_host));

    STRONG_ASSERT(t1.num_elements()==t2.num_elements());

    t1.store(d1, to_host);
    t2.store(d2, to_host);

    float res = 0;

    for ( long_t i = 0; i < t1.num_elements(); ++i )
    {
        res = std::max(res, std::abs(d1[i]-d2[i]));
    }

    detail::tensor::free(d1, from_host);
    detail::tensor::free(d2, from_host);

    return res;
}

template<typename Net>
inline void compare_conv( long_t n, long_t l1, long_t l2,
                          vec3i const & is, vec3i const & ws,
                          host_tensor<float,5> & kernels,
                          host_tensor<float,1> & biases,
                          host_tensor<float,5> & input,
                          host_tensor<float,5> & output )
{
    Net net(n,l1,l2,is,ws,kernels.data(),biases.data());
    device_tensor<float,5> in(n,l1,is[0],is[1],is[2]);
    in = input;

    std::cout << "Net needs: "
              << (net.resident_memory() / 1024 / 1024)
              << " MB of Resident memory, and "
              << (net.working_memory() / 1024 / 1024)
              << " MB of Working memory\n";

    auto out = net.forward(std::move(in));

    std::cout << "YAYA: " << max_abs_diff(output,out) << std::endl;
}

template<typename Net>
inline void compare_mfp_pool( long_t n, long_t l,
                              vec3i const & is, vec3i const & ws,
                              host_tensor<float,5> & input,
                              host_tensor<float,5> & output )
{
    Net net(n,l,is,ws);
    device_tensor<float,5> in(n,l,is[0],is[1],is[2]);
    in = input;

    auto out = net.forward(std::move(in));

    std::cout << "YAYA: " << max_abs_diff(output,out) << std::endl;
}

// template<typename Net>
// inline void compare_pool( long_t n, long_t l,
//                           vec3i const & is, vec3i const & ws,
//                           host_tensor<float,5> & input,
//                           host_tensor<float,5> & output )
// {
//     Net net(n,l,is,ws);
//     host_tensor<float,5> in(n,l,is[0],is[1],is[2]);
//     in = input;

//     auto out = net.forward(std::move(in));

//     std::cout << "YAYA: " << max_abs_diff(output,out) << std::endl;
// }


void conv_test()
{
    static std::mt19937 rng = std::mt19937(1234);

    std::uniform_int_distribution<long_t> intdist(2,5);
    vec3i ws(intdist(rng),intdist(rng),intdist(rng));

    std::uniform_int_distribution<long_t> intdist2(2,24);
    vec3i os(intdist2(rng),intdist2(rng),intdist2(rng));

    std::uniform_int_distribution<long_t> intdist3(1,25);

    long_t n = intdist3(rng);
    long_t l1 = intdist3(rng);
    long_t l2 = intdist3(rng);

    // long_t n = 8;
    // long_t l1 = 64;
    // long_t l2 = 48;

    vec3i is = os + ws - vec3i::one;

    host_tensor<float,5> kernels(rand_init,l1,l2,ws[0],ws[1],ws[2]);
    host_tensor<float,1> biases(rand_init,l2);

    host::v1::naive_conv net(n,l1,l2,is,ws,kernels.data(),biases.data());
    host_tensor<float,5> input(rand_init,n,l1,is[0],is[1],is[2]);

    host_tensor<float,5> in1(n,l1,is[0],is[1],is[2]);
    in1 = input;

    auto output = net.forward(std::move(in1));

    compare_conv<device::v1::cudnn_conv>
        (n,l1,l2,is,ws,kernels,biases,input,output);

    compare_conv<device::v1::cudnn_no_precomp_gemm_conv>
        (n,l1,l2,is,ws,kernels,biases,input,output);

    compare_conv<device::v1::fft_conv>
        (n,l1,l2,is,ws,kernels,biases,input,output);

    std::cout << "\n\n";
}



void mfp_test()
{
    static std::mt19937 rng = std::mt19937(1234);

    std::uniform_int_distribution<long_t> intdist(1,4);
    vec3i ws(intdist(rng),intdist(rng),intdist(rng));

    std::uniform_int_distribution<long_t> intdist2(2,13);
    vec3i os(intdist2(rng),intdist2(rng),intdist2(rng));

    std::uniform_int_distribution<long_t> intdist3(1,5);

    long_t n = intdist3(rng);
    long_t l = intdist3(rng);

    vec3i is;
    is[0] = (ws[0] > 1) ? ws[0] * os[0] + ws[0] - 1: os[0];
    is[1] = (ws[1] > 1) ? ws[1] * os[1] + ws[1] - 1: os[1];
    is[2] = (ws[2] > 1) ? ws[2] * os[2] + ws[2] - 1: os[2];

    host::v1::naive_boost_mfp net(n,l,is,ws);
    host_tensor<float,5> input(rand_init,n,l,is[0],is[1],is[2]);

    host_tensor<float,5> in1(n,l,is[0],is[1],is[2]);
    in1 = input;

    auto output = net.forward(std::move(in1));

    compare_mfp_pool<device::v1::cudnn_mfp>(n,l,is,ws,input,output);
//     compare_mfp<host::v1::mfp>(n,l,is,ws,input,output);

    std::cout << "\n\n";
}

void pool_test()
{
    static std::mt19937 rng = std::mt19937(1234);

    std::uniform_int_distribution<long_t> intdist(1,4);
    vec3i ws(intdist(rng),intdist(rng),intdist(rng));

    std::uniform_int_distribution<long_t> intdist2(2,13);
    vec3i os(intdist2(rng),intdist2(rng),intdist2(rng));

    std::uniform_int_distribution<long_t> intdist3(1,5);

    long_t n = intdist3(rng);
    long_t l = intdist3(rng);

    vec3i is = os * ws;

    host::v1::naive_boost_pool net(n,l,is,ws);
    host_tensor<float,5> input(rand_init,n,l,is[0],is[1],is[2]);

    host_tensor<float,5> in1(n,l,is[0],is[1],is[2]);
    in1 = input;

    auto output = net.forward(std::move(in1));

    compare_mfp_pool<device::v1::cudnn_pool>(n,l,is,ws,input,output);

    std::cout << "\n\n";
}

int main()
{
    while (1)
    {
        conv_test();
        mfp_test();
        pool_test();
    }
}
