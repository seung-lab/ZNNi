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
                      vec3i(5,55,55), vec3i(1,3,3),
                      conv2a_k, conv2a_b)));

  // conv2b
  float conv2b_k[36*36*1*3*3];
  float conv2b_b[36];
  read_from_file<float>("./0412_VD2D3D-MS/conv2b/filters",conv2b_k,36*36*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv2b/biases",conv2b_b,36);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (4, 36, 36,
                      vec3i(5,53,53), vec3i(1,3,3),
                      conv2b_k, conv2b_b)));

  // pool2 using mfp
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_mfp
                    (4, 36, vec3i(5,51,51), vec3i(1,2,2))));

  // conv3a
  float conv3a_k[36*48*1*3*3];
  float conv3a_b[48];
  read_from_file<float>("./0412_VD2D3D-MS/conv3a/filters",conv3a_k,36*48*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv3a/biases",conv3a_b,48);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 36, 48,
                      vec3i(5,25,25), vec3i(1,3,3),
                      conv3a_k, conv3a_b)));
  // conv3b
  float conv3b_k[48*48*1*3*3];
  float conv3b_b[48];
  read_from_file<float>("./0412_VD2D3D-MS/conv3b/filters",conv3b_k,48*48*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv3b/biases",conv3b_b,48);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (16, 48, 48,
                      vec3i(5,23,23), vec3i(1,3,3),
                      conv3b_k, conv3b_b)));

  // pool3 using mfp
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_mfp
                    (16, 48, vec3i(5,21,21), vec3i(1,2,2))));

  // conv4a
  float conv4a_k[48*60*1*3*3];
  float conv4a_b[60];
  read_from_file<float>("./0412_VD2D3D-MS/conv4a/filters",conv4a_k,48*60*1*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv4a/biases",conv4a_b,60);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (64, 48, 60,
                      vec3i(5,10,10), vec3i(1,3,3),
                      conv4a_k, conv4a_b)));

  // conv4b-p1
  float conv4b_p1_k[60*60*2*3*3];
  float conv4b_p1_b[60];
  read_from_file<float>("./0412_VD2D3D-MS/conv4b-p1/filters",conv4b_p1_k,48*60*2*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv4b-p1/biases",conv4b_p1_b,60);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (64, 48, 60,
                      vec3i(5,8,8), vec3i(2,3,3),
                      conv4b_p1_k, conv4b_p1_b)));
  // pool4-p1 : maxfilter
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                   (new device::v1::cudnn_maxfilter
                    (64, 48, vec3i(4,6,6), vec3i(2,2,2))));
  // conv5a-p1
  float conv5a_p1_k[60*60*2*3*3];
  float conv5a_p1_b[60];
  read_from_file<float>("./0412_VD2D3D-MS/conv5a-p1/filters",conv5a_k,60*60*2*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv5a-p1/biases",conv5a_b,60);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (64, 60, 60,
                      vec3i(3,5,5), vec3i(2,3,3),
                      conv5a_p1_k, conv5a_p1_b)));

  // conv5b
  float conv5b_p1_k[60*72*2*3*3];
  float conv5b_p1_b[72];
  read_from_file<float>("./0412_VD2D3D-MS/conv5b-p1/filters",conv5b_p1_k,60*72*2*3*3);
  read_from_file<float>("./0412_VD2D3D-MS/nconv5b-p1/biases",conv5b_p1_b,72);
  layers.push_back(std::unique_ptr<device::v1::device_layer>
                    (new device::v1::cudnn_no_precomp_gemm_conv
                     (64, 60, 72,
                      vec3i(2,3,3), vec3i(2,3,3),
                      conv5b_p1_k, conv5b_p1_b)));
}

}} // namespace znn::fwd
