#include "znn/util/deshuffler.hpp"
#include "znn/util/network.hpp"

#include "znn/network/n4_gpu.hpp"
#include "znn/network/multiscale_gpu_b1.hpp"
#include "znn/network/multiscale_gpu_b2.hpp"
#include "znn/network/multiscale_gpu_b3.hpp"
#include "znn/util/dataprovider.hpp"

#include <zi/time.hpp>

using namespace znn::fwd;

int main(int argc, char *argv[])
{
  // record time
  zi::wall_timer timer;
  timer.reset();

  // settings
  vec3i outsz(1,8,8);
  h5vec3 fov(9, 109, 109);
  h5vec3 h5outsz(outsz[0], outsz[1], outsz[2]);

  // The magnificent dataprovider
  DataProvider dp(h5outsz, fov);
  if (argc >= 4)
	  dp.LoadHDF(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]));
  else {
	  std::cout << "Usage: znni inputfile.h5 outputfile.h5 datasetname\n";
	  return -1;
  }

  // Create layers for all three multiscale branches
  auto layers_b1 = create_multiscale_b1(outsz);
  auto layers_b2 = create_multiscale_b2(outsz);
  auto layers_b3 = create_multiscale_b3(outsz);

  // Everyday I'm shufflin'
  deshuffler deshuffler_b1(vec3i(1, 8, 8));
  deshuffler_b1.split(vec3i(1, 2, 2));
  deshuffler_b1.split(vec3i(1, 2, 2));
  deshuffler_b1.split(vec3i(1, 2, 2));

  deshuffler deshuffler_b2(vec3i(1, 8, 8));
  deshuffler_b2.split(vec3i(1, 2, 2));
  deshuffler_b2.split(vec3i(1, 2, 2));

  deshuffler deshuffler_b3(vec3i(1, 8, 8));
  deshuffler_b3.split(vec3i(1, 2, 2));



  // intermediate variables
  device_tensor<float, 5> b1, b2, b3;
  device_tensor<float, 5> out_patch(1, 3, outsz[0], outsz[1], outsz[2]);

  // iterate all the patches
  for (auto it = dp.begin(); it!=dp.end(); ++it) {
    b1 = dp.ReadWindowData(*it, to_device);
    b2 = dp.ReadWindowData(*it, to_device); // FIXME: Optimize with copy assignment for tensors?
    b3 = dp.ReadWindowData(*it, to_device);

    for (auto & l: layers_b1) {
      b1 = l->forward(std::move(b1));
    }
    for (auto & l: layers_b2) {
      b2 = l->forward(std::move(b2));
    }
    for (auto & l: layers_b3) {
      b3 = l->forward(std::move(b3));
    }

    // Deshuffle
    host_tensor<float, 5> output_b1(64, 1, outsz[0], outsz[1] / 8, outsz[2] / 8);
    host_tensor<float, 5> output_b2(16, 1, outsz[0], outsz[1] / 4, outsz[2] / 4);
    host_tensor<float, 5> output_b3(4, 1, outsz[0], outsz[1] / 2, outsz[2] / 2);

    output_b1.load(b1.data(), from_device);
    output_b2.load(b2.data(), from_device);
    output_b3.load(b3.data(), from_device);

    host_array<real> deshuffled_b1 = deshuffler_b1.deshuffle(output_b1.data());
    host_array<real> deshuffled_b2 = deshuffler_b2.deshuffle(output_b2.data());
    host_array<real> deshuffled_b3 = deshuffler_b3.deshuffle(output_b3.data());

    std::transform(deshuffled_b1.begin(), deshuffled_b1.end(), deshuffled_b2.begin(), deshuffled_b1.begin(), std::plus<real>());
    std::transform(deshuffled_b1.begin(), deshuffled_b1.end(), deshuffled_b3.begin(), deshuffled_b1.begin(), std::plus<real>());


    std::cout << "Processing took: " << timer.elapsed<double>() << "\n";
    timer.reset();

    // push to data provider
    //dp.WriteWindowData(*it, host_out_patch);
    std::cout << "push to data provider: " << timer.elapsed<double>() << "\n";
  }
}
