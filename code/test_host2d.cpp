#include "znn/tensor/tensor.hpp"
#include "znn/host/2d/naive_boost_conv.hpp"
#include "znn/host/2d/naive_conv.hpp"
//#include "znn/host/2d/fft_conv_serial.hpp"
#include "znn/host/2d/fft_conv.hpp"
//#include "znn/host/2d/dp_fft_conv.hpp"
//#include "znn/host/2d/direct_conv.hpp"
#include "znn/host/2d/naive_boost_mfp.hpp"
#include "znn/host/2d/naive_mfp.hpp"
#include "znn/host/2d/mfp.hpp"
//#include "znn/host/2d/naive_boost_pool.hpp"
//#include "znn/host/2d/naive_pool.hpp"
//#include "znn/host/2d/pool.hpp"
//#include "znn/util/network.hpp"


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
                          vec2i const & is, vec2i const & ws,
                          host_tensor<float,4> & kernels,
                          host_tensor<float,1> & biases,
                          host_tensor<float,4> & input,
                          host_tensor<float,4> & output )
{
    Net net(n,l1,l2,is,ws,kernels.data(),biases.data());
    host_tensor<float,4> in(n,l1,is[0],is[1]);
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
inline void compare_mfp( long_t n, long_t l,
                         vec2i const & is, vec2i const & ws,
                         host_tensor<float,4> & input,
                         host_tensor<float,4> & output )
{
    Net net(n,l,is,ws);
    host_tensor<float,4> in(n,l,is[0],is[1]);
    in = input;

    auto out = net.forward(std::move(in));

    std::cout << "YAYA: " << max_abs_diff(output,out) << std::endl;
}

// template<typename Net>
// inline void compare_pool( long_t n, long_t l,
//                           vec2i const & is, vec2i const & ws,
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

    std::uniform_int_distribution<long_t> intdist(2,9);
    vec2i ws(intdist(rng),intdist(rng));


    std::uniform_int_distribution<long_t> intdist2(2,24);
    vec2i os(intdist2(rng),intdist2(rng));

    std::uniform_int_distribution<long_t> intdist3(1,15);

    long_t n = intdist3(rng);
    long_t l1 = intdist3(rng);
    long_t l2 = intdist3(rng);

    // long_t n = 8;
    // long_t l1 = 64;
    // long_t l2 = 48;

    vec2i is = os + ws - vec2i::one;

    host_tensor<float,4> kernels(rand_init,l1,l2,ws[0],ws[1]);
    host_tensor<float,1> biases(rand_init,l2);

    host::twod::naive_conv2d net(n,l1,l2,is,ws,kernels.data(),biases.data());
    host_tensor<float,4> input(rand_init,n,l1,is[0],is[1]);

    host_tensor<float,4> in1(n,l1,is[0],is[1]);
    in1 = input;

    auto output = net.forward(std::move(in1));

    compare_conv<host::twod::naive_boost_conv2d>
        (n,l1,l2,is,ws,kernels,biases,input,output);

    compare_conv<host::twod::fft_conv2d>
         (n,l1,l2,is,ws,kernels,biases,input,output);

    // compare_conv<host::2d::fft_conv>
    //     (n,l1,l2,is,ws,kernels,biases,input,output);

    // compare_conv<host::2d::fft_conv_serial>
    //     (n,l1,l2,is,ws,kernels,biases,input,output);

    // compare_conv<host::2d::dp_fft_conv>
    //     (n,l1,l2,is,ws,kernels,biases,input,output);

    // compare_conv<host::2d::direct_conv>
    //     (n,l1,l2,is,ws,kernels,biases,input,output);

    // compare_conv<device::ram::ram_conv<device::ram::gemm_it>>
    //     (n,l1,l2,is,ws,kernels,biases,input,output);

    // compare_conv<device::ram::ram_conv<device::ram::fft_it>>
    //     (n,l1,l2,is,ws,kernels,biases,input,output);


    std::cout << "\n\n";
}



void mfp_test()
{
    static std::mt19937 rng = std::mt19937(1234);

    std::uniform_int_distribution<long_t> intdist(1,4);
    vec2i ws(intdist(rng),intdist(rng));

    std::uniform_int_distribution<long_t> intdist2(2,13);
    vec2i os(intdist2(rng),intdist2(rng));

    std::uniform_int_distribution<long_t> intdist3(1,5);

    long_t n = intdist3(rng);
    long_t l = intdist3(rng);

    vec2i is;
    is[0] = (ws[0] > 1) ? ws[0] * os[0] + ws[0] - 1: os[0];
    is[1] = (ws[1] > 1) ? ws[1] * os[1] + ws[1] - 1: os[1];

    host::twod::naive_mfp2d net(n,l,is,ws);
    host_tensor<float,4> input(rand_init,n,l,is[0],is[1]);

    host_tensor<float,4> in1(n,l,is[0],is[1]);
    in1 = input;

    auto output = net.forward(std::move(in1));

    compare_mfp<host::twod::naive_boost_mfp2d>(n,l,is,ws,input,output);
    compare_mfp<host::twod::mfp2d>(n,l,is,ws,input,output);

    std::cout << "\n\n";
}

// void pool_test()
// {
//     static std::mt19937 rng = std::mt19937(1234);

//     std::uniform_int_distribution<long_t> intdist(1,4);
//     vec2i ws(intdist(rng),intdist(rng),intdist(rng));

//     std::uniform_int_distribution<long_t> intdist2(2,13);
//     vec2i os(intdist2(rng),intdist2(rng),intdist2(rng));

//     std::uniform_int_distribution<long_t> intdist3(1,5);

//     long_t n = intdist3(rng);
//     long_t l = intdist3(rng);

//     vec2i is = os * ws;

//     host::2d::naive_pool net(n,l,is,ws);
//     host_tensor<float,5> input(rand_init,n,l,is[0],is[1],is[2]);

//     host_tensor<float,5> in1(n,l,is[0],is[1],is[2]);
//     in1 = input;

//     auto output = net.forward(std::move(in1));

//     compare_pool<host::2d::naive_pool>(n,l,is,ws,input,output);
//     compare_pool<host::2d::pool>(n,l,is,ws,input,output);
//     compare_pool<host::2d::pool>(n,l,is,ws,input,output);


//     std::cout << "\n\n";
//}

int main()
{
    while (1)
    {
        conv_test();
        mfp_test();
        //pool_test();
    }
}
