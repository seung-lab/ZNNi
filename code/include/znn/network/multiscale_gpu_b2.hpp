#pragma once

#include "znn/util/io.hpp"
#include "znn/device/v1/cudnn_mfp.hpp"
#include "znn/device/v1/cudnn_crop.hpp"
#include "znn/device/v1/cudnn_no_precomp_gemm_conv.hpp"
#include "znn/device/v1/cudnn_maxfilter.hpp"
#include "znn/tensor/tensor.hpp"

#include <string>

namespace znn {namespace fwd{

std::vector<std::unique_ptr<device::v1::device_layer>>
create_multiscale_b2(const vec3i & outsz)
{
  std::vector<std::unique_ptr<device::v1::device_layer>> layers;

  // crop
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_crop
                    (1, 24, vec3i(9,116,116), vec3i(1,18,18))));

  // conv1a
  float conv1a_k[1*24*1*3*3];
  float conv1a_b[24];
  read_from_file<float>("./0412_VD2D3D-MS/conv1a/filters",conv1a_k,1*24*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv1a/biases",conv1a_b,24);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_no_precomp_gemm_conv
                    (1, 1, 24,
                     vec3i(7,80,80), vec3i(1,3,3),
                     conv1a_k, conv1a_b, activation::relu)));

  // conv1b
  float conv1b_k[24*24*3*3];
  float conv1b_b[24];
  read_from_file<float>("./0412_VD2D3D-MS/conv1b/filters",conv1b_k,24*24*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv1b/biases",conv1b_b,24);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_no_precomp_gemm_conv
                    (1, 24, 24,
                     vec3i(7,78,78), vec3i(1,3,3),
                     conv1b_k, conv1b_b, activation::relu)));

  // conv1c
  float conv1c_k[24*24*2*2];
  float conv1c_b[24];
  read_from_file<float>("./0412_VD2D3D-MS/conv1c/filters",conv1c_k,24*24*1*2*2);
  read_from_file<float>("./0412_VD2D3D-MS/nconv1c/biases",conv1c_b,24);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_no_precomp_gemm_conv
                    (1, 24, 24,
                     vec3i(7,76,76), vec3i(1,2,2),
                     conv1c_k, conv1c_b, activation::relu)));

  // pool1 using mfp
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_mfp
                    (1, 24, vec3i(7,75,75), vec3i(1,2,2))));

  // conv2a
  float conv2a_k[24*36*1*3*3];
  float conv2a_b[36];
  read_from_file<float>("./0412_VD2D3D-MS/conv2a/filters",conv2a_k,24*36*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv2a/biases",conv2a_b,36);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (4, 24, 36,
                      vec3i(7,37,37), vec3i(1,3,3),
                      conv2a_k, conv2a_b, activation::relu)));

  // conv2b
  float conv2b_k[36*36*1*3*3];
  float conv2b_b[36];
  read_from_file<float>("./0412_VD2D3D-MS/conv2b/filters",conv2b_k,36*36*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv2b/biases",conv2b_b,36);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (4, 36, 36,
                      vec3i(7,35,35), vec3i(1,3,3),
                      conv2b_k, conv2b_b, activation::relu)));

  // pool2 using mfp
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_mfp
                    (4, 36, vec3i(7,33,33), vec3i(1,2,2))));

  // conv3a
  float conv3a_k[36*48*1*3*3];
  float conv3a_b[48];
  read_from_file<float>("./0412_VD2D3D-MS/conv3a/filters",conv3a_k,36*48*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv3a/biases",conv3a_b,48);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 36, 48,
                      vec3i(7,16,16), vec3i(1,3,3),
                      conv3a_k, conv3a_b, activation::relu)));
  // conv3b
  float conv3b_k[48*48*1*3*3];
  float conv3b_b[48];
  read_from_file<float>("./0412_VD2D3D-MS/conv3b/filters",conv3b_k,48*48*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv3b/biases",conv3b_b,48);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 48, 48,
                      vec3i(7,14,14), vec3i(1,3,3),
                      conv3b_k, conv3b_b, activation::relu)));

  // pool3-p2: max filtering
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_mfp
                    (16, 48, vec3i(7,12,12), vec3i(2,2,2))));

  // conv4a-p2
  float conv4a_p2_k[48*48*2*3*3];
  float conv4a_p2_b[48];
  read_from_file<float>("./0412_VD2D3D-MS/conv4a-p2/filters",conv4a_p2_k,48*60*2*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv4a-p2/biases",conv4a_p2_b,48);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 48, 48,
                      vec3i(6,11,11), vec3i(2,3,3),
                      conv4a_p2_k, conv4a_p2_b, activation::relu)));

  // conv4b-p2
  float conv4b_p2_k[48*48*2*3*3];
  float conv4b_p2_b[48];
  read_from_file<float>("./0412_VD2D3D-MS/conv4b-p2/filters",conv4b_p2_k,48*48*2*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv4b-p2/biases",conv4b_p2_b,48);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 48, 48,
                      vec3i(5,9,9), vec3i(2,3,3),
                      conv4b_p2_k, conv4b_p2_b, activation::relu)));
  // pool4-p2 : maxfilter
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_maxfilter
                    (16, 48, vec3i(4,7,7), vec3i(2,2,2))));
  // conv5a-p2
  float conv5a_p2_k[48*60*2*3*3];
  float conv5a_p2_b[60];
  read_from_file<float>("./0412_VD2D3D-MS/conv5a-p2/filters",conv5a_p2_k,48*60*2*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv5a-p2/biases",conv5a_p2_b,60);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 48, 60,
                      vec3i(3,6,6), vec3i(2,3,3),
                      conv5a_p2_k, conv5a_p2_b, activation::relu)));

  // conv5b-p2
  float conv5b_p2_k[60*72*2*3*3];
  float conv5b_p2_b[72];
  read_from_file<float>("./0412_VD2D3D-MS/conv5b-p2/filters",conv5b_p2_k,60*72*2*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv5b-p2/biases",conv5b_p2_b,72);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 60, 72,
                      vec3i(2,4,4), vec3i(2,3,3),
                      conv5b_p2_k, conv5b_p2_b, activation::relu)));

  return layers;
}

}} // namespace znn::fwd
