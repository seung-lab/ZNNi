#include "znn/util/deshuffler.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/device/v1/cudnn_conv.hpp"
#include "znn/device/v1/cudnn_mfp.hpp"
#include "znn/device/v1/cudnn_maxfilter.hpp"
#include "znn/device/v1/cudnn_crop.hpp"


#include <string>
#include <fstream>
#include <sstream>
#include <vector>

#include <zi/time.hpp>

using namespace znn::fwd;

void xxxxx( vec3i const & is, long_t rounds )
{
    vec3i  fov(9,113,113);
    vec3i  sz = is + fov - vec3i::one;
    long_t bs = 1;

    //////////////////
    // Path 123a
    std::vector<std::unique_ptr<device::v1::device_layer>> path123a;

    path123a.push_back(make_unique<device::v1::cudnn_conv>  ( bs,  1,  24, sz, vec3i(1,3,3)));
    sz = sz - vec3i(1,3,3) + vec3i::one;

    path123a.push_back(make_unique<device::v1::cudnn_conv>  ( bs, 24,  24, sz, vec3i(1,3,3)));
    sz = sz - vec3i(1,3,3) + vec3i::one;

    path123a.push_back(make_unique<device::v1::cudnn_conv>  ( bs, 24,  24, sz, vec3i(1,2,2)));
    sz = sz - vec3i(1,2,2) + vec3i::one;

    path123a.push_back(make_unique<device::v1::cudnn_mfp>   ( bs, 24,  sz, vec3i(1,2,2)));
    sz /= vec3i(1,2,2);  bs *= 4;

    path123a.push_back(make_unique<device::v1::cudnn_conv>  ( bs, 24,  36, sz, vec3i(1,3,3)));
    sz = sz - vec3i(1,3,3) + vec3i::one;

    path123a.push_back(make_unique<device::v1::cudnn_conv>  ( bs, 36,  36, sz, vec3i(1,3,3)));
    sz = sz - vec3i(1,3,3) + vec3i::one;

    std::cout << "Path 123a: " << sz << ' ' << bs << "\n";

    vec3i  tmpsz = sz;
    long_t tmpbs = bs;

    //////////////////
    // Path 1b
    std::vector<std::unique_ptr<device::v1::device_layer>> path1b;

    path1b.push_back(make_unique<device::v1::cudnn_crop>     ( bs,  36, sz, vec3i(0,17,17)));
    sz = sz - vec3i(0,34,34);

    path1b.push_back(make_unique<device::v1::cudnn_maxfilter>( bs,  36, sz, vec3i(1,2,2)));
    sz = sz - vec3i(1,2,2) + vec3i::one;

    path1b.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  36, 36, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path1b.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  36, 36, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path1b.push_back(make_unique<device::v1::cudnn_maxfilter>( bs,  36, sz, vec3i(2,2,2)));
    sz = sz - vec3i(2,2,2) + vec3i::one;

    path1b.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  36, 48, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path1b.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  48, 48, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path1b.push_back(make_unique<device::v1::cudnn_maxfilter>( bs,  48, sz, vec3i(2,2,2)));
    sz = sz - vec3i(2,2,2) + vec3i::one;

    path1b.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  48, 60, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path1b.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  60, 60, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path1b.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  60,200, sz, vec3i(1,1,1)));
    sz = sz - vec3i(1,1,1) + vec3i::one;

    vec3i  p1bsz = sz;
    long_t p1bbs = bs;

    std::cout << "Path 1b: " << sz << ' ' << bs << "\n";

    sz = tmpsz;
    bs = tmpbs;

    //////////////////
    // Path 23b
    std::vector<std::unique_ptr<device::v1::device_layer>> path23b;

    path23b.push_back(make_unique<device::v1::cudnn_crop>     ( bs,  36, sz, vec3i(1,0,0)));
    sz = sz - vec3i(2,0,0);

    path23b.push_back(make_unique<device::v1::cudnn_mfp>      ( bs,  36, sz, vec3i(1,2,2)));
    sz /= vec3i(1,2,2);  bs *= 4;

    path23b.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  36, 48, sz, vec3i(1,3,3)));
    sz = sz - vec3i(1,3,3) + vec3i::one;

    path23b.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  48, 48, sz, vec3i(1,3,3)));
    sz = sz - vec3i(1,3,3) + vec3i::one;

    tmpsz = sz;
    tmpbs = bs;

    std::cout << "Path 23b: " << sz << ' ' << bs << "\n";


    //////////////////
    // Path 2c
    std::vector<std::unique_ptr<device::v1::device_layer>> path2c;

    path2c.push_back(make_unique<device::v1::cudnn_crop>     ( bs,  48, sz, vec3i(0,5,5)));
    sz = sz - vec3i(0,10,10);

    path2c.push_back(make_unique<device::v1::cudnn_maxfilter>( bs,  48, sz, vec3i(2,2,2)));
    sz = sz - vec3i(2,2,2) + vec3i::one;

    path2c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  48, 48, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path2c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  48, 48, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path2c.push_back(make_unique<device::v1::cudnn_maxfilter>( bs,  48, sz, vec3i(2,2,2)));
    sz = sz - vec3i(2,2,2) + vec3i::one;

    path2c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  48, 60, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path2c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  60, 60, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    path2c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  60, 200, sz, vec3i(1,1,1)));
    sz = sz - vec3i(1,1,1) + vec3i::one;

    std::cout << "Path 2c: " << sz << ' ' << bs << "\n";

    sz = tmpsz;
    bs = tmpbs;

    //////////////////
    // Path 3c
    std::vector<std::unique_ptr<device::v1::device_layer>> path3c;

    path3c.push_back(make_unique<device::v1::cudnn_crop>     ( bs,  48, sz, vec3i(1,0,0)));
    sz = sz - vec3i(2,0,0);

    std::cout << "Path 3c: " << sz << ' ' << bs << "\n";

    path3c.push_back(make_unique<device::v1::cudnn_mfp>      ( bs, 48,  sz, vec3i(1,2,2)));
    sz /= vec3i(1,2,2);  bs *= 4;

    std::cout << "Path 3c: " << sz << ' ' << bs << "\n";

    path3c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  48, 60, sz, vec3i(1,3,3)));
    sz = sz - vec3i(1,3,3) + vec3i::one;

    std::cout << "Path 3c: " << sz << ' ' << bs << "\n";

    path3c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  48, 48, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    std::cout << "Path 3c: " << sz << ' ' << bs << "\n";

    path3c.push_back(make_unique<device::v1::cudnn_maxfilter>( bs,  48, sz, vec3i(2,2,2)));
    sz = sz - vec3i(2,2,2) + vec3i::one;

    std::cout << "Path 3c: " << sz << ' ' << bs << "\n";

    path3c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  48, 60, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    std::cout << "Path 3c: " << sz << ' ' << bs << "\n";

    path3c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  60, 60, sz, vec3i(2,3,3)));
    sz = sz - vec3i(2,3,3) + vec3i::one;

    std::cout << "Path 3c: " << sz << ' ' << bs << "\n";

    path3c.push_back(make_unique<device::v1::cudnn_conv>     ( bs,  60, 200, sz, vec3i(1,1,1)));
    sz = sz - vec3i(1,1,1) + vec3i::one;

    std::cout << "Path 3c: " << sz << ' ' << bs << "\n";

}


int main()
{
    xxxxx(vec3i(1,8,8),1);
}
