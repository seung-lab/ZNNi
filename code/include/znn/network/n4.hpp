#include "znn/util/io.hpp"
#include "znn/host/v1/mfp.hpp"
#include "znn/host/v1/fft_conv.hpp"
#include "znn/host/v1/dp_fft_conv.hpp"
#include "znn/host/v1/fft_conv.hpp"
#include "znn/host/v1/maxout.hpp"
#include "znn/tensor/tensor.hpp"

#include <string>

namespace znn {namespace fwd{

std::vector<std::unique_ptr<host::v1::host_layer>>
create_n4(const vec3i outsz)
{
  std::vector<std::unique_ptr<host::v1::host_layer>> layers;
  
  // conv1
  float conv1_k[1*48*4*4];
  float conv1_b[48];
  read_from_file<float>("./zfish-N4_A/conv1/kernels",conv1_k,1*48*1*4*4);
  read_from_file<float>("./zfish-N4_A/nconv1/biases",conv1_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::fft_conv
                    (1, 48, 48,
                     vec3i(1,95,95)+outsz-vec3i(1,1,1), vec3i(1,4,4),
                     conv1_k, conv1_b)));

  // pool1 using mfp
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::mfp
                    (1, 48, vec3i(1,92,92)+outsz-vec3i(1,1,1), vec3i(1,2,2))));
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::maxout(4, 48, 2, vec3i(1,46,46)+outsz-vec3i(1,1,1))));

  // conv2
  float conv2_k[48*48*5*5];
  float conv2_b[48];
  read_from_file<float>("./zfish-N4_A/conv2/filters",conv2_k,48*48*1*5*5);
  read_from_file<float>("./zfish-N4_A/nconv2/biases",conv2_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::fft_conv
                    (4, 48, 48,
                     vec3i(1,46,46)+outsz-vec3i(1,1,1), vec3i(1,5,5),
                     conv2_k, conv2_b)));

  // pool2 using mfp
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::mfp
                    (4, 48, vec3i(1,42,42)+outsz-vec3i(1,1,1), vec3i(1,2,2))));
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::maxout
                    (16, 48, 2, vec3i(1,21,21)+outsz-vec3i(1,1,1))));

  // conv3
  float conv3_k[48*48*1*4*4];
  float conv3_b[48];
  read_from_file<float>("./zfish-N4_A/conv3/filters",conv3_k,48*48*1*4*4);
  read_from_file<float>("./zfish-N4_A/nconv3/biases",conv3_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                    (new host::v1::fft_conv
                     (16, 48, 48,
                      vec3i(1,21,21)+outsz-vec3i(1,1,1), vec3i(1,4,4),
                      conv3_k, conv3_b)));

  // pool3 using mfp
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::mfp
                    (16, 48, vec3i(1,18,18)+outsz-vec3i(1,1,1), vec3i(1,2,2))));
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::maxout
                    (64, 48, 2, vec3i(1,9,9)+outsz-vec3i(1,1,1))));

  // conv4
  float conv4_k[48*48*1*4*4];
  float conv4_b[48];
  read_from_file<float>("./zfish-N4_A/conv4/filters",conv4_k,48*48*1*4*4);
  read_from_file<float>("./zfish-N4_A/nconv4/biases",conv4_b,48);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                    (new host::v1::fft_conv
                     (64, 48, 48,
                      vec3i(1,9,9)+outsz-vec3i(1,1,1), vec3i(1,4,4),
                      conv4_k, conv4_b)));

  // pool4 using mfp
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::mfp
                    (64, 48, vec3i(1,6,6)+outsz-vec3i(1,1,1), vec3i(1,2,2))));
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                   (new host::v1::maxout
                    (256, 48, 2, vec3i(1,3,3)+outsz-vec3i(1,1,1))));

  // conv5
  float conv5_k[48*200*1*3*3];
  float conv5_b[200];
  read_from_file<float>("./zfish-N4_A/conv5/filters",conv5_k,48*200*1*3*3);
  read_from_file<float>("./zfish-N4_A/nconv5/biases",conv5_b,200);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                    (new host::v1::fft_conv
                     (256, 48, 200,
                      vec3i(1,3,3)+outsz-vec3i(1,1,1), vec3i(1,3,3),
                      conv5_k, conv5_b)));

  // conv6
  float conv6_k[200*3*1*1*1];
  float conv6_b[3];
  read_from_file<float>("./zfish-N4_A/conv6/filters",conv6_k,200*3*1*1*1);
  read_from_file<float>("./zfish-N4_A/output/biases",conv6_b,3);
  layers.push_back(std::unique_ptr<host::v1::host_layer>
                    (new host::v1::fft_conv
                     (256, 200, 3,
                      vec3i(1,1,1)+outsz-vec3i(1,1,1), vec3i(1,1,1),
                      conv6_k, conv6_b)));
  std::cout<< "finish reading layers!"<< "\n";
  return layers;
}

  }} // namespace znn::fwd