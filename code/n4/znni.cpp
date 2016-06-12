#include "io.hpp"
#include "znn/util/network.hpp"
#include "znn/host/v1/mfp.hpp"
#include "znn/host/v1/fft_conv.hpp"
#include "znn/host/v1/pool.hpp"
#include "znn/host/v1/dp_fft_conv.hpp"
#include "znn/host/v1/direct_conv.hpp"

#include <zi/time.hpp>

#include <string>
#include <fstream>
#include <sstream>

using namespace znn::fwd;

std::string net_name;
vec3i       os;
long_t      max_memory = static_cast<long_t>(240) * 1024 * 1024 * 1024; // GB

int main(int argc, char *argv[])
{
  std::vector<std::unique_ptr<host::v1::host_layer>> layers;
  
  // conv1
  float conv1_k[64*32*4*4];
  float conv1_b[64];
  read_from_file<float>("./exper_aleks/conv1.kernels",conv1_k,3*48*4*4);
  read_from_file<float>("./exper_aleks/conv1.biases",conv1_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::direct_conv
                    (48, 48, 48,
                     vec3i(1,201,201), vec3i(1,4,4),
                     conv1_k, conv1_b)));

  //pool1
  layers.push_back(std::unique_ptr<host::v1::host_layer> (new host::v1::pool(48, 48,
                                       vec3i(1,198,198), vec3i(1,2,2))));

  // conv2
  float conv2_k[64*32*4*4];
  float conv2_b[64];
  read_from_file<float>("./exper_aleks/conv2.kernels",conv2_k,3*48*4*4);
  read_from_file<float>("./exper_aleks/conv2.biases",conv2_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::direct_conv
                    (48, 48, 48,
                     vec3i(1,99,99), vec3i(1,5,5),
                     conv1_k, conv1_b)));


  //pool2
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::pool(48, 48,
                                       vec3i(1,95,95), vec3i(1,2,2))));

  // conv3
  float conv3_k[64*32*4*4];
  float conv3_b[64];
  read_from_file<float>("./exper_aleks/conv2.kernels",conv3_k,3*48*4*4);
  read_from_file<float>("./exper_aleks/conv2.biases",conv3_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                    (new host::v1::direct_conv
                     (48, 48, 48,
                      vec3i(1,47,47), vec3i(1,4,4),
                      conv3_k, conv3_b)));


  //pool3
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::pool(48, 48,
                                       vec3i(1,44,44), vec3i(1,2,2))));

  // conv4
  float conv4_k[64*32*4*4];
  float conv4_b[64];
  read_from_file<float>("./exper_aleks/conv2.kernels",conv4_k,3*48*4*4);
  read_from_file<float>("./exper_aleks/conv2.biases",conv4_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                    (new host::v1::direct_conv
                     (48, 48, 48,
                      vec3i(1,22,22), vec3i(1,4,4),
                      conv4_k, conv4_b)));
  
  //pool4
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::pool(48, 48,
                                       vec3i(1,11,11), vec3i(1,2,2))));

  // conv5
  float conv5_k[64*32*4*4];
  float conv5_b[64];
  read_from_file<float>("./exper_aleks/conv2.kernels",conv5_k,3*48*4*4);
  read_from_file<float>("./exper_aleks/conv2.biases",conv5_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                    (new host::v1::direct_conv
                     (48, 48, 48,
                      vec3i(1,5,5), vec3i(1,3,3),
                      conv5_k, conv5_b)));

  // conv6
  float conv6_k[64*32*4*4];
  float conv6_b[64];
  read_from_file<float>("./exper_aleks/conv2.kernels",conv6_k,3*48*4*4);
  read_from_file<float>("./exper_aleks/conv2.biases",conv6_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                    (new host::v1::direct_conv
                     (200, 48*200, 200*3,
                      vec3i(1,5,5), vec3i(1,1,1),
                      conv6_k, conv6_b)));

  // output

 
}
