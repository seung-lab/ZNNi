#include "znn/util/deshuffler.hpp"
#include "znn/util/network.hpp"

#include "znn/network/n4.hpp"
#include "znn/util/dataprovider.hpp"

#include <zi/time.hpp>

using namespace znn::fwd;

int main(int argc, char *argv[])
{
  vec3i outsz(1,100,100)
  // create layers for n4 network
  auto layers = create_n4();

  // record time
  zi::wall_timer wt;
  wt.reset();
  // data provider here
  h5vec3 fov(1, 89, 89);
  h5vec3 output(1,100,100);
  DataProvider dp(output, fov);
  if (argc >= 4)
    ok = dp.LoadHDF(std::string(argv[1], std::string(argv[2]), std::string(argv[3])));
  else
    std::cout<< "Usage: znni inputfile.h5 outputfile.h5 datasetname\n";
  if (!ok) return -1;

  // prepare variables for iteration
  h5vec3 dimensions;
  hsize_t elcnt = output.x() * output.y() * output.z();
  float * outpatch = new float[3*elcnt];
  h5vec3 halffov((fov-1)/2);
  h5vec3 input(output + fov - 1);
  hsize_t w, inz, iny, inx, outz, outy, outx;

  // run forward pass for one patch
  tensor<float,5> in;
  tensor<float,5> out;

  // shuffler
  deshuffler ds(vec3i(1,976,976));
  ds.split(vec3i(1,2,2));
  ds.split(vec3i(1,2,2));
  ds.split(vec3i(1,2,2));

  // iterate all the patches
  for (auto it = dp.begin(); it!=dp.end(); ++it){
    std::unique_ptr<float[]> inpatch = dp.ReadWindowData(*it, dimensions);
    outpatch = forward(layers, inpatch);
    for (auto & l: layers){
      auto out = l.forward(in);
      in = out;
    }

    tensor<float, 5> hresult(256, 1, 1, 100, 100);
    for (long_t i=0; i<256; ++i){
      hresult[i][0] = out[i][0]
    }
    std::cout << "Processing took: " << wt. elapsed<double>() << "\n";
    wt.reset();

    auto rr = ds.deshuffle(hresult.data());
    wt.reset();
    // push to data provider
    1;95;0c
    ////////
    std::cout << "push to data provider: " << wt.elapsed<double>() << "\n";
  }
}
