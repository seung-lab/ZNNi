#include "znn/util/deshuffler.hpp"
#include "znn/util/dataprovider.hpp"
#include "znn/util/znnhelper.hpp"

#include "znn/host/v1/fft_conv.hpp"

#include "znn/network/multiscale_cpu_b1.hpp"
#include "znn/network/multiscale_cpu_b2.hpp"
#include "znn/network/multiscale_cpu_b3.hpp"

#include <zi/time.hpp>

#include <functional> // for summation of final layers

using namespace znn::fwd;

int main(int argc, char *argv[])
{
  // record time
  zi::wall_timer timer;
  timer.reset();

  // settings
  vec3i outsz(17, 136, 136); // must be multiple of 8!
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
  read_from_file<float>("./0421_VD2D3D-MS/convout/filters", convout_k, 200 * 3 * 1 * 1 * 1);
  fix_dims(convout_k, 200, 3, 1, 1, 1);
  read_from_file<float>("./0421_VD2D3D-MS/output/biases", convout_b, 3);
  host::v1::fft_conv final_conv(1, 200, 3, vec3i(1, 8, 8), vec3i(1, 1, 1), convout_k, convout_b, activation::sigmoid);

  // Write sum of all three branches to b1 + BIAS
  std::array<float, 200> convx_b;
  read_from_file<float>("./0421_VD2D3D-MS/nconvx/biases", convx_b.data(), 200);

  // Everyday I'm shufflin'
  deshuffler deshuffler_b1(outsz);
  deshuffler_b1.split(vec3i(1, 2, 2));

  deshuffler deshuffler_b2(outsz);
  deshuffler_b2.split(vec3i(1, 2, 2));
  deshuffler_b2.split(vec3i(1, 2, 2));

  deshuffler deshuffler_b3(outsz);
  deshuffler_b3.split(vec3i(1, 2, 2));
  deshuffler_b3.split(vec3i(1, 2, 2));
  deshuffler_b3.split(vec3i(1, 2, 2));

  // intermediate variables
  host_tensor<float, 5> b1, b2, b3;
  host_tensor<float, 5> out_patch(1, 3, outsz[0], outsz[1], outsz[2]);

  // iterate all the patches
  for (auto it = dp.begin(); it!=dp.end(); ++it) {
    b1 = dp.ReadWindowData(*it, to_host);
    b2 = dp.ReadWindowData(*it, to_host);
    b3 = dp.ReadWindowData(*it, to_host);

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

    host_tensor<float, 5> single_output_b1(4, 1, outsz[0], outsz[1] / 2, outsz[2] / 2);
    host_tensor<float, 5> single_output_b2(16, 1, outsz[0], outsz[1] / 4, outsz[2] / 4);
    host_tensor<float, 5> single_output_b3(64, 1, outsz[0], outsz[1] / 8, outsz[2] / 8);

    host_tensor<float, 5> out_patch(1, 200, outsz[0], outsz[1], outsz[2]);
    host_tensor<float, 5> out_patch_b2(1, 200, outsz[0], outsz[1], outsz[2]);
    host_tensor<float, 5> out_patch_b3(1, 200, outsz[0], outsz[1], outsz[2]);

    for (int ch = 0; ch < 200; ++ch) {
      for (int n = 0; n < 4; ++n) {
        single_output_b1[n][0] = b1[n][ch];
      }
      for (int n = 0; n < 16; ++n) {
        single_output_b2[n][0] = b2[n][ch];
      }
      for (int n = 0; n < 64; ++n) {
        single_output_b3[n][0] = b3[n][ch];
      }
      out_patch[0][ch].load(deshuffler_b1.deshuffle(single_output_b1.data()).data(), from_host);
      out_patch_b2[0][ch].load(deshuffler_b2.deshuffle(single_output_b2.data()).data(), from_host);
      out_patch_b3[0][ch].load(deshuffler_b3.deshuffle(single_output_b3.data()).data(), from_host);

      std::transform(out_patch[0][ch].begin(), out_patch[0][ch].end(), out_patch_b2[0][ch].begin(), out_patch[0][ch].begin(), std::plus<real>());
      std::transform(out_patch[0][ch].begin(), out_patch[0][ch].end(), out_patch_b3[0][ch].begin(), out_patch[0][ch].begin(), std::plus<real>());
      std::for_each(out_patch[0][ch].begin(), out_patch[0][ch].end(), [&convx_b,ch](float& val) { val += convx_b[ch]; });

    }

    relu(out_patch.data(), 200 * outsz[0] * outsz[1] * outsz[2]);

    auto affinity = final_conv.forward(std::move(out_patch));

    std::cout << "Processing took: " << timer.elapsed<double>() << "\n";
    timer.reset();

    dp.WriteWindowData(*it, affinity);
  }
}
