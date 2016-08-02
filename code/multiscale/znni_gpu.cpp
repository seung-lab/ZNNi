#include "znn/util/deshuffler.hpp"
#include "znn/util/dataprovider.hpp"
#include "znn/util/znnhelper.hpp"

#include "znn/network/multiscale_gpu_b1.hpp"
#include "znn/network/multiscale_gpu_b2.hpp"
#include "znn/network/multiscale_gpu_b3.hpp"

#include "znn/device/common/kernels.hpp"
#include "znn/device/v1/cudnn_activation.hpp"

#include <zi/time.hpp>

#include <boost/lexical_cast.hpp> // converting arguments

using namespace znn::fwd;

int main(int argc, char *argv[])
{
  // record time
  zi::wall_timer timer, timer_all;
  timer_all.reset();
  timer.reset();

  // settings
  vec3i outsz(12,128,128); // must be multiple of 8!
  if (argc == 7) {
    try {
      outsz = vec3i(boost::lexical_cast<int>(argv[4]),
                    boost::lexical_cast<int>(argv[5]),
                    boost::lexical_cast<int>(argv[6]));
    }
    catch(boost::bad_lexical_cast) {
      std::cout << "Can't read output size parameters.\n";
    }
  }
  h5vec3 fov(9, 109, 109);
  h5vec3 h5outsz(outsz[0], outsz[1], outsz[2]);

  // The magnificent dataprovider
  DataProvider dp(h5outsz, fov);
  if (argc >= 4) {
    if (!dp.LoadHDF(std::string(argv[1]), std::string(argv[2]), std::string(argv[3]))) {
      std::cout << "Could not initialize dataprovider.\n";
      return -1;
    }
  }
  else {
    std::cout << "Usage: znni inputfile.h5 outputfile.h5 datasetname [12 128 128]\n";
    return -1;
  }

  // Create layers for all three multiscale branches
  auto layers_b1 = create_multiscale_b1(outsz);
  auto layers_b2 = create_multiscale_b2(outsz);
  auto layers_b3 = create_multiscale_b3(outsz);

  // Create final convolution layers
  float convout_k[200 * 3 * 1 * 1 * 1];
  float convout_b[3];
  read_from_file<float>("./VD2D3D-MS/convout/filters",convout_k,200*3*1*1*1);
  fix_dims(convout_k, 200, 3, 1, 1, 1);
  read_from_file<float>("./VD2D3D-MS/output/biases", convout_b, 3);
  device::v1::cudnn_conv final_conv(1, 200, 3, outsz, vec3i(1, 1, 1), convout_k, convout_b, activation::sigmoid);

  // Write sum of all three branches and biases to branch 1
  float convx_b[200];
  read_from_file<float>("./VD2D3D-MS/nconvx/biases", convx_b, 200);
  device::v1::cudnn_activation relu(1, 200, outsz, convx_b, activation::relu);

  // intermediate variables
  device_tensor<float, 5> b1;//, b2, b3;
  device_tensor<float, 5> out_patch(1, 3, outsz[0], outsz[1], outsz[2]);

  // iterate all the patches
  int active_patch = 1;
  int num_patches = dp.size();
  for (auto it = dp.begin(); it!=dp.end(); ++it) {
    timer.reset();

    b1 = dp.ReadWindowData(*it, to_device);

    device_tensor<float, 5> b2(1, 1, b1.shape()[2], b1.shape()[3], b1.shape()[4]);
    device_tensor<float, 5> b3(1, 1, b1.shape()[2], b1.shape()[3], b1.shape()[4]);
    b2.load(b1.data(), from_device); // faster than reading it from dp again
    b3.load(b1.data(), from_device);

    std::cout << timer.elapsed<double>() << "s for reading input from disk\n";

    timer.reset();
    try { // if we run out of GPU memory during forward pass, then somewhere here
      for (auto & l: layers_b1) {
        b1 = l->forward(std::move(b1));
      }

      for (auto & l: layers_b2) {
        b2 = l->forward(std::move(b2));
      }
      device::add_to(b2.data(), b2.data() + b2.num_elements(), b1.data(), 1.f); // Add B2 output to B1
      b2.reset();

      for (auto & l: layers_b3) {
        b3 = l->forward(std::move(b3));
      }
      device::add_to(b3.data(), b3.data() + b3.num_elements(), b1.data(), 1.f); // Add B3 output to B1
      b3.reset();
    }
    catch (std::logic_error e) {
      std::cout << "Error: " << e.what() << "\n Aborting program...\n";
      return -2;
    }
    catch (...) {
      std::cout << "Unknown Error during forward pass. Aborting program...\n";
      return -3;
    }
    std::cout << timer.elapsed<double>() << "s for main branches\n";

    timer.reset();
    b1 = relu.forward(std::move(b1)); // Relu Activation
    b1 = final_conv.forward(std::move(b1));
    std::cout << timer.elapsed<double>() << "s for relu activation function + final convolution\n";

    timer.reset();
    host_tensor<float, 5> affinity(1, 3, outsz[0], outsz[1], outsz[2]);
    affinity.load(b1.data(), from_device);

    dp.WriteWindowData(*it, affinity);
    std::cout << timer.elapsed<double>() << "s for loading result from GPU and writing to disk\n";

    b1.reset();
    std::cout << timer_all.elapsed<double>() << "s in total for patch " << active_patch++ << "/" << num_patches <<"!\n";
  }

  std::cout << "Processing succesfully completed after " << timer_all.elapsed<double>() << "s.\n";
  return 0;
}
