#include "znn/util/deshuffler.hpp"
#include "znn/util/network.hpp"

#include "znn/network/n4.hpp"
#include "znn/util/dataprovider.hpp"

#include <zi/time.hpp>

using namespace znn::fwd;

int main(int argc, char *argv[])
{
  vec3i outsz(1,100,100);
  // create layers for n4 network
  auto layers = create_n4(outsz);

  // record time
  zi::wall_timer wt;
  wt.reset();
  // data provider here
  h5vec3 fov(1, 85, 85);
  h5vec3 h5outsz(1,100,100);
  DataProvider dp(h5outsz, fov);
  if (argc >= 4)
    dp.LoadHDF(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]));
  else{
    std::cout<< "Usage: znni inputfile.h5 outputfile.h5 datasetname\n";
    return -1;
  }

  // run forward pass for one patch
  host_tensor<float,5> out_patch(1,3,1,100,100);

  // shuffler
  deshuffler ds(vec3i(1,976,976));
  ds.split(vec3i(1,2,2));
  ds.split(vec3i(1,2,2));
  ds.split(vec3i(1,2,2));

  // iterate all the patches
  for (auto it = dp.begin(); it!=dp.end(); ++it){
    host_tensor<float, 5> in_patch = dp.ReadWindowData(*it);
    for (auto & l: layers){
      auto out = l->forward(std::move(in_patch));
      in_patch = out;
    }

    host_tensor<float, 5> hresult(256, 1, 1, 100, 100);
    for (long_t i=0; i<256; ++i){
      hresult[i][0] = out_patch[i][0];
    }
    std::cout << "Processing took: " << wt. elapsed<double>() << "\n";
    wt.reset();

    host_array<real> rr = ds.deshuffle(hresult.data());
    wt.reset();
    // push to data provider
    dp.WriteWindowData(*it, rr.data());
    ////////
    std::cout << "push to data provider: " << wt.elapsed<double>() << "\n";
  }
}
