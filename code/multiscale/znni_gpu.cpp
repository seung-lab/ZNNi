#include "znn/util/deshuffler.hpp"
#include "znn/util/dataprovider.hpp"
#include "znn/util/znnhelper.hpp"

#include "znn/network/multiscale_gpu_b1.hpp"
#include "znn/network/multiscale_gpu_b2.hpp"
#include "znn/network/multiscale_gpu_b3.hpp"

#include "znn/device/common/kernels.hpp"
#include "znn/device/v1/cudnn_activation.hpp"

#include <zi/time.hpp>

#include <functional> // for summation of final layers

using namespace znn::fwd;

int main(int argc, char *argv[])
{
  // record time
  zi::wall_timer timer, timer_all;
  timer_all.reset();
  timer.reset();

  // settings
  vec3i outsz(2,16,16); // must be multiple of 8!
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

  // Create final convolution layers
  float convout_k[200 * 3 * 1 * 1 * 1];
  float convout_b[3];
  read_from_file<float>("./0421_VD2D3D-MS/convout/filters",convout_k,200*3*1*1*1);
  fix_dims(convout_k, 200, 3, 1, 1, 1);
  read_from_file<float>("./0421_VD2D3D-MS/output/biases", convout_b, 3);
  device::v1::cudnn_no_precomp_gemm_conv final_conv(1, 200, 3, outsz, vec3i(1, 1, 1), convout_k, convout_b, activation::sigmoid);

  // Write sum of all three branches and biases to branch 1
  float convx_b[200];
  read_from_file<float>("./0421_VD2D3D-MS/nconvx/biases", convx_b, 200);
  device::v1::cudnn_activation relu(1, 200, outsz, convx_b, activation::relu);

  // intermediate variables
  device_tensor<float, 5> b1, b2, b3;
  device_tensor<float, 5> out_patch(1, 3, outsz[0], outsz[1], outsz[2]);

  // iterate all the patches
  int active_patch = 1;
  for (auto it = dp.begin(); it!=dp.end(); ++it) {
    timer_all.reset();
    timer.reset();
    b1 = dp.ReadWindowData(*it, to_device);
    b2 = dp.ReadWindowData(*it, to_device); // FIXME: Optimize with copy assignment for tensors?
    b3 = dp.ReadWindowData(*it, to_device);
    std::cout << timer.elapsed<double>() << "s for reading input from disk\n";

    timer.reset();
    for (auto & l: layers_b1) {
      b1 = l->forward(std::move(b1));
    }
    for (auto & l: layers_b2) {
      b2 = l->forward(std::move(b2));
    }
    for (auto & l: layers_b3) {
      b3 = l->forward(std::move(b3));
    }
    std::cout << timer.elapsed<double>() << "s for main branches and GPU deshuffling\n";

    timer.reset();
#if 0
    device::add_to(b1.data(), b1.data() + b1.num_elements(), b2.data(), 1.f); // Add B2 output to B1
    device::add_to(b1.data(), b1.data() + b1.num_elements(), b3.data(), 1.f); // Add B3 output to B1
    std::cout << timer.elapsed<double>() << "s for summation of branches (GPU side)\n";
#else
    host_tensor<float, 5> out_patch(1, 200, outsz[0], outsz[1], outsz[2]);
    host_tensor<float, 5> out_patch_b2(1, 200, outsz[0], outsz[1], outsz[2]);
    host_tensor<float, 5> out_patch_b3(1, 200, outsz[0], outsz[1], outsz[2]);
    out_patch.load(b1.data(), from_device);
    out_patch_b2.load(b2.data(), from_device);
    out_patch_b3.load(b3.data(), from_device);

    for (int ch = 0; ch < 200; ++ch) {
      std::transform(out_patch[0][ch].begin(), out_patch[0][ch].end(), out_patch_b2[0][ch].begin(), out_patch[0][ch].begin(), std::plus<real>());
      std::transform(out_patch[0][ch].begin(), out_patch[0][ch].end(), out_patch_b3[0][ch].begin(), out_patch[0][ch].begin(), std::plus<real>());
      //std::for_each(out_patch[0][ch].begin(), out_patch[0][ch].end(), [&convx_b, ch](float& val) { val += convx_b[ch]; });
    }
    b1.load(out_patch.data(), from_host);
    std::cout << timer.elapsed<double>() << "s for summation of branches (GPU->CPU->GPU)\n";
#endif
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
    b2.reset();
    b3.reset();
    std::cout << timer_all.elapsed<double>() << "s in total for this patch!\n";
  }
}
