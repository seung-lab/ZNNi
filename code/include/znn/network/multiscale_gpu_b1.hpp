#pragma once

#include "znn/util/io.hpp"
#include "znn/device/v1/cudnn_mfp.hpp"
#include "znn/device/v1/cudnn_no_precomp_gemm_conv.hpp"
#include "znn/device/v1/maxout.hpp"
#include "znn/tensor/tensor.hpp"

#include <string>

namespace znn {namespace fwd{

std::vector<std::unique_ptr<device::v1::device_layer>>
create_multiscale_b1(const vec3i outsz)
{
  std::vector<std::unique_ptr<device::v1::device_layer>> layers;

  // conv1a
  float conv1a_k[1*24*1*3*3];
  float conv1a_b[24];
  read_from_file<float>("./0412_VD2D3D-MS/conv1a/filters",conv1a_k,1*24*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv1a/biases",conv1a_b,24);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_no_precomp_gemm_conv
                    (1, 1, 24,
                     vec3i(5,116,116), vec3i(1,3,3),
                     conv1a_k, conv1a_b)));

  // conv1b
  float conv1b_k[24*24*3*3];
  float conv1b_b[24];
  read_from_file<float>("./0412_VD2D3D-MS/conv1b/filters",conv1b_k,24*24*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv1b/biases",conv1b_b,24);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_no_precomp_gemm_conv
                    (1, 24, 24,
                     vec3i(5,114,114), vec3i(1,3,3),
                     conv1b_k, conv1b_b)));

  // conv1c
  float conv1c_k[24*24*2*2];
  float conv1c_b[24];
  read_from_file<float>("./0412_VD2D3D-MS/conv1c/filters",conv1c_k,24*24*1*2*2);
  read_from_file<float>("./0412_VD2D3D-MS/nconv1c/biases",conv1c_b,24);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_no_precomp_gemm_conv
                    (1, 24, 24,
                     vec3i(5,112,112), vec3i(1,2,2),
                     conv1c_k, conv1c_b)));

  // pool1 using mfp
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_mfp
                    (1, 24, vec3i(5,111,111), vec3i(1,2,2))));

  // conv2a
  float conv2a_k[24*36*1*3*3];
  float conv2a_b[36];
  read_from_file<float>("./0412_VD2D3D-MS/conv2a/filters",conv2a_k,24*36*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv2a/biases",conv2a_b,36);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (4, 24, 36,
                      vec3i(5,56,56), vec3i(1,3,3),
                      conv2a_k, conv2a_b)));

  // conv2b
  float conv2b_k[36*36*1*3*3];
  float conv2b_b[36];
  read_from_file<float>("./0412_VD2D3D-MS/conv2b/filters",conv2b_k,36*36*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv2b/biases",conv2b_b,36);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (4, 36, 36,
                      vec3i(5,54,54), vec3i(1,3,3),
                      conv2b_k, conv2b_b)));

  // pool2 using mfp
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_mfp
                    (4, 36, vec3i(5,52,52), vec3i(1,2,2))));

  // conv3a
  float conv3a_k[36*48*1*3*3];
  float conv3a_b[48];
  read_from_file<float>("./0412_VD2D3D-MS/conv3a/filters",conv3a_k,36*48*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv3a/biases",conv3a_b,48);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 36, 48,
                      vec3i(5,26,26), vec3i(1,3,3),
                      conv3a_k, conv3a_b)));
  // conv3b
  float conv3b_k[48*48*1*3*3];
  float conv3b_b[48];
  read_from_file<float>("./0412_VD2D3D-MS/conv3b/filters",conv3b_k,48*48*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv3b/biases",conv3b_b,48);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 48, 48,
                      vec3i(5,24,24), vec3i(1,3,3),
                      conv3b_k, conv3b_b)));

  // pool3 using mfp
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_mfp
                    (16, 48, vec3i(5,22,22), vec3i(1,2,2))));

  // conv4a
  float conv4a_k[48*60*1*3*3];
  float conv4a_b[60];
  read_from_file<float>("./0412_VD2D3D-MS/conv4a/filters",conv4a_k,48*60*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv4a/biases",conv4a_b,60);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (64, 48, 60,
                      vec3i(5,11,11), vec3i(1,3,3),
                      conv4a_k, conv4a_b)));

  // conv4b
  float conv4b_k[60*60*2*3*3];
  float conv4b_b[60];
  read_from_file<float>("./0412_VD2D3D-MS/conv4b/filters",conv4b_k,48*60*2*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv4b/biases",conv4b_b,60);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (64, 48, 60,
                      vec3i(5,11,11), vec3i(2,3,3),
                      conv4b_k, conv4b_b)));

  // conv6
  float conv6_k[200*3*1*1*1];
  float conv6_b[3];
  read_from_file<float>("./0412_VD2D3D-MS/conv6/filters",conv6_k,200*3*1*1*1);
  read_from_file<float>("./0412_VD2D3D-MS/output/biases",conv6_b,3);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (256, 200, 3,
                      vec3i(1,1,1), vec3i(1,1,1),
                      conv6_k, conv6_b)));
  std::cout<< "finish reading layers!"<< "\n";
  return layers;
}

  }} // namespace znn::fwd
