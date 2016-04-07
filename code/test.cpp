#include "znn/device/fusion/network.hpp"


#include "znn/log.hpp"
#include "znn/device/tail/sub_network.hpp"
#include "znn/device/tail/network.hpp"
#include "znn/device/ram/network.hpp"
#include "znn/util/network_data.hpp"

#include <string>
#include <fstream>
#include <sstream>

#include <zi/time.hpp>

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

std::string net_name;
vec3i       os;
long_t      max_memory = static_cast<long_t>(240) * 1024 * 1024 * 1024; // GB

int main(int argc, char *argv[])
{
    // device::ram::in_out_split_conv<device::ram::native_cudnn_conv>
    //     ntr(10,2,20,2,vec3i(14,14,14),vec3i(3,3,3));

    // std::cout << ntr.workspace_size() << std::endl;

    net_name = std::string(argv[1]);
    std::string net_path    = "../networks/" + net_name + ".znni";
    std::string report_path = "../reports/" + net_name + ".best_device.report";

    network_descriptor ndesc(net_path);
    network_data nd(ndesc, 1, vec3i(16,16,16));

    device::ram::network<device::ram::fft_it> net1(nd,2);
    device::ram::network<device::ram::fft_it> net2(nd,4);
    device::ram::network<device::ram::fft_it> net3(nd,6);
    std::cout << net1.memory_required() << std::endl;
    std::cout << net2.memory_required() << std::endl;
    std::cout << net3.memory_required() << std::endl;

    host_tensor<float,5> in1(rand_init,net1.in_shape());
    host_tensor<float,5> in2(net1.in_shape());
    host_tensor<float,5> in3(net2.in_shape());

    in2 = in1;
    in3 = in1;

    zi::wall_timer wt1;
    in1 = net1.forward(std::move(in1));
    std::cout << wt1.elapsed<double>() << "   <---\n";

    zi::wall_timer wt2;
    in2 = net2.forward(std::move(in2));
    std::cout << wt2.elapsed<double>() << "   <---\n";

    zi::wall_timer wt3;
    in3 = net3.forward(std::move(in3));
    std::cout << wt3.elapsed<double>() << "   <---\n";

    std::cout << "DIFF: " << max_abs_diff(in1,in2) << "\n";
    std::cout << "DIFF: " << max_abs_diff(in1,in3) << "\n";

    // {
    //     auto n = device::tail::network<device::tail::fft_conv>::get(nd);
    //     //device::tail::sub_network<device::tail::fft_conv> snt(nd);
    //     std::cout << n->memory_requirement() << "\n";
    // }

    // {
    //     auto n = device::tail::network<device::tail::cudnn_conv>::get(nd);
    //     std::cout << n->memory_requirement() << "\n";
    //     //device::tail::sub_network<device::tail::cudnn_conv> snt(nd);
    //     //std::cout << snt.memory_requirement() << "\n";
    // }


}
