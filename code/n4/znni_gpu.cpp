#include "znn/util/deshuffler.hpp"
#include "znn/util/network.hpp"

#include "znn/network/n4_gpu.hpp"
#include "znn/util/dataprovider.hpp"

#include <zi/time.hpp>

using namespace znn::fwd;

int main(int argc, char *argv[])
{
  vec3i outsz(1,16,16);
  // create layers for n4 network
  auto layers = create_n4(outsz);
  std::cout<<"layers created!"<<std::endl;

  // record time
  zi::wall_timer wt;
  wt.reset();
  // data provider here
  h5vec3 fov(1, 95, 95);
  h5vec3 h5outsz(outsz[0], outsz[1], outsz[2]);
  DataProvider dp(h5outsz, fov);
  if (argc >= 4)
    dp.LoadHDF(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]));
  else{
    std::cout<< "Usage: znni inputfile.h5 outputfile.h5 datasetname\n";
    return -1;
  }

  // shuffler
  deshuffler ds(vec3i(1,16,16));
  ds.split(vec3i(1,2,2));
  ds.split(vec3i(1,2,2));
  ds.split(vec3i(1,2,2));
	ds.split(vec3i(1,2,2));

  // intermediate variables
  device_tensor<float, 5> inout;
  device_tensor<float,5> out_patch(1,3, outsz[0], outsz[1], outsz[2]);

  // iterate all the patches
  for (auto it = dp.begin(); it!=dp.end(); ++it){
    inout = dp.ReadWindowData(*it);
    int li = 0;
    for (auto & l: layers){
      std::cout<<"layer: "<< ++li<<std::endl;
      inout = l->forward(std::move(inout));
    }

    host_tensor<float, 5> hresult(256, 1, 1, 1, 1);
    for (long_t i=0; i<256; ++i){
      hresult[i][0] = inout[i][0];
    }
    std::cout << "Processing took: " << wt.elapsed<double>() << "\n";
    wt.reset();

    host_array<real> rr = ds.deshuffle(hresult.data());
    wt.reset();

    host_tensor<float, 5> host_out_patch(1, 3, outsz[0], outsz[1], outsz[2]);
    host_out_patch.load(rr.data(), from_device);
    // push to data provider
    dp.WriteWindowData(*it, host_out_patch);
    ////////
    std::cout << "push to data provider: " << wt.elapsed<double>() << "\n";
  }
}
