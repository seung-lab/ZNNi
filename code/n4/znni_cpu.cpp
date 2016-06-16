#include "znn/util/deshuffler.hpp"
#include "znn/util/network.hpp"

#include "znn/network/n4_cpu.hpp"
#include "znn/util/dataprovider.hpp"

#include <zi/time.hpp>

using namespace znn::fwd;

int main(int argc, char *argv[])
{

  vec3i outsz(1,16,16); //vec3i outsz(1,100,100);
  // create layers for n4 network
  auto layers = create_n4(outsz);
  std::cout<<"layers created!"<<std::endl;

  // record time
  zi::wall_timer wt;
  wt.reset();
  // data provider here
  h5vec3 fov(1, 95, 95);
  h5vec3 h5outsz(outsz[0], outsz[1], outsz[2]); //h5vec3 h5outsz(1, 100, 100);
  DataProvider dp(h5outsz, fov);
  if (argc >= 4)
    dp.LoadHDF(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]));
  else{
    std::cout<< "Usage: znni inputfile.h5 outputfile.h5 datasetname\n";
    return -1;
  }

  // shuffler
  deshuffler ds(vec3i(1,976,976));
  ds.split(vec3i(1,2,2));
  ds.split(vec3i(1,2,2));
  ds.split(vec3i(1,2,2));

  // intermediate variables
	host_tensor<float, 5> inout;// (256, 48, 1, 194, 194);
  host_tensor<float,5> out_patch(1,3, outsz[0], outsz[1], outsz[2]); //host_tensor<float, 5> out_patch(1, 3, 1, 100, 100);

  // iterate all the patches
  for (auto it = dp.begin(); it!=dp.end(); ++it){
    inout = dp.ReadWindowData(*it);
    //std::cout<<"shape of input patch: "<< inout.shape_vec()<<std::endl;
    /*for (int i=0; i<184*184; i++)
        std::cout<<in_patch.data()[i]<<", ";*/
    int li = 0;
    for (auto & l: layers){
      std::cout<<"layer: "<< ++li<<std::endl;
      inout = l->forward(std::move(inout));
    }

    host_tensor<float, 5> hresult(256, 1, outsz[0], outsz[1], outsz[2]); //host_tensor<float, 5> hresult(256, 1, 1, 100, 100);
    for (long_t i=0; i<256; ++i){
      hresult[i][0] = inout[i][0];
    }
    std::cout << "Processing took: " << wt. elapsed<double>() << "\n";
    wt.reset();

    host_array<real> rr = ds.deshuffle(hresult.data());
    wt.reset();
    out_patch.load(rr.data(), from_host);
    // push to data provider
    dp.WriteWindowData(*it, out_patch);
    ////////
    std::cout << "push to data provider: " << wt.elapsed<double>() << "\n";
  }
}
