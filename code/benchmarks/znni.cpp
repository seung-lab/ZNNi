#include "znn/util/deshuffler.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/direct_conv.hpp"
#include "znn/host/v1/mfp.hpp"
#include "znn/host/v1/maxout.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include <zi/time.hpp>

using namespace znn::fwd;

template < typename T >
inline bool read_from_file( const std::string& fname, T* data, std::size_t n )
{
    FILE* f = std::fopen(fname.c_str(), "rbXS");
    if ( !f ) return false;

    std::size_t nread = std::fread(data, sizeof(T), n, f);
    std::fclose(f);

    std::cout << "NREAD: " << nread << std::endl;

    return nread == n;
}

template < typename T >
inline bool
write_to_file( const std::string& fname,
               const T* data, std::size_t n )
{
    std::ofstream f(fname.c_str(), (std::ios::out | std::ios::binary) );
    if ( !f ) return false;

    f.write( reinterpret_cast<const char*>(data), n * sizeof(T));
    return true;
}


void xxxxx()
{
    float k[64*32*4*4];
    float b[64];

    std::vector<std::unique_ptr<host::v1::host_layer>> layers;

    read_from_file<float>("./exper_aleks/layer1.kernels",k,3*64*4*4);
    read_from_file<float>("./exper_aleks/layer1.biases",b,64);

    layers.push_back(make_unique<host::v1::direct_conv>(1,3,64,vec2i(1028,1028),vec2i(4,4),k,b));
    layers.push_back(make_unique<host::v1::maxout>(1,64,2,vec2i(1025,1025)));
    layers.push_back(make_unique<host::v1::mfp>(1,32,vec2i(1025,1025),vec2i(2,2)));

    read_from_file<float>("./exper_aleks/layer2.kernels",k,32*64*4*4);
    read_from_file<float>("./exper_aleks/layer2.biases",b,64);
    layers.push_back(make_unique<host::v1::direct_conv>(4,32,64,vec2i(512,512),vec2i(4,4),k,b));
    layers.push_back(make_unique<host::v1::mfp>(4,64,vec2i(509,509),vec2i(2,2)));
    layers.push_back(make_unique<host::v1::maxout>(16,64,2,vec2i(254,254)));

    read_from_file<float>("./exper_aleks/layer3.kernels",k,32*64*4*4);
    read_from_file<float>("./exper_aleks/layer3.biases",b,64);
    layers.push_back(make_unique<host::v1::direct_conv>(16,32,64,vec2i(254,254),vec2i(4,4),k,b));
    // layers.push_back(make_unique<host::v1::mfp>(16,64,vec2i(251,251),vec2i(2,2)));
    // layers.push_back(make_unique<host::v1::maxout>(64,64,2,vec2i(125,125)));

    layers.push_back(make_unique<host::v1::maxout>(16,64,2,vec2i(251,251)));
    layers.push_back(make_unique<host::v1::mfp>(16,32,vec2i(251,251),vec2i(2,2)));


    read_from_file<float>("./exper_aleks/layer4.kernels",k,32*4*4*4);
    read_from_file<float>("./exper_aleks/layer4.biases",b,4);
    layers.push_back(make_unique<host::v1::direct_conv>(64,32,4,vec2i(125,125),vec2i(4,4),k,b));

    host_array<float> input(1028*1028*256);

    {
        zi::wall_timer wt;
        wt.reset();

        STRONG_ASSERT(read_from_file<float>("./exper_aleks/data.raw",
                                            input.data(),
                                            1028*1028*256));

        std::cout << "Read all data took: " << wt.elapsed<double>() << std::endl;
    }

    long_t workspace_size = 0;
    long_t inout_size = 0;
    for ( auto & l: layers )
    {
      inout_size = std::max(inout_size, l->input_memory);
      inout_size = std::max(inout_size, l->output_memory);
      workspace_size = std::max(workspace_size, l->total_memory());
    }
    host_array<float> inout[2];
    inout[0] = host_array<float>(rand_init,inout_size/4);
    inout[1] = host_array<float>(rand_init,inout_size/4);
    host_array<char> wspace(workspace_size);

    deshuffler ds(vec3i(1,976,976));
    ds.split(vec3i(1,2,2));
    ds.split(vec3i(1,2,2));
    ds.split(vec3i(1,2,2));

    host_tensor<float,4> hresultx(64,2,122,122);
    host_tensor<float,4> hresult(64,1,122,122);

    for ( long_t stages = 0; stages < 254; ++stages )
    {

        std::cout << "Processing slice no " << (stages+1) << std::endl;

        zi::wall_timer wt;
        wt.reset();

        inout[0].load_n(input.data() + 1028*1028*stages, 1028*1028*3, from_host);

        long_t lnum = 0;
        for ( auto & l: layers )
        {
            l->forward(inout[lnum%2].data(), inout[(lnum+1)%2].data(), wspace.data());
            ++lnum;
        }

        //host_tensor_ref<float,4> dresult(inout[lnum%2].data(), 64, 4, 122, 122);
        hresultx.load(inout[lnum%2].data(), from_host);

        for ( long_t i = 0; i < 64; ++i )
        {
            hresult[i][0] = hresultx[i][0];
        }

        std::cout << "Processing took: " << wt.elapsed<double>() << "\n";
        wt.reset();

        std::string outfile = "/tmp/slices/slice_";
        if ( stages < 9  ) outfile += "0";
        if ( stages < 99 ) outfile += "0";

        outfile += std::to_string(stages+1) + ".raw";

        //write_to_file("./exper_aleks/resx.raw", hresult.data(), 122*122*64);

        auto rr = ds.deshuffle(hresult.data());

        wt.reset();
        write_to_file(outfile, rr.data(), 976*976);

        std::cout << "Saving took: " << wt.elapsed<double>() << "\n";
    }
}


int main()
{
    xxxxx();
}
