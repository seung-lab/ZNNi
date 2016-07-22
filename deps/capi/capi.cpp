#include "znn/util/znnhelper.hpp"
#include "znn/device/v1/cudnn_mfp.hpp"
#include "znn/device/v1/cudnn_crop.hpp"
#include "znn/device/v1/cudnn_conv.hpp"
#include "znn/device/v1/cudnn_maxfilter.hpp"
#include "znn/device/v1/cudnn_assemble.hpp"
#include "znn/tensor/tensor.hpp"

#include <string>

#include "capi.h"

using namespace znn::fwd;

struct znnilayer{
    std::unique_ptr<device::v1::device_layer> layer;
};

znnilayer* crop_layer(int ni, int no, int* insz, int* outsz){
    znnilayer *l = new znnilayer();
    l->layer = std::unique_ptr<znn::fwd::device::v1::device_layer>(
                                new device::v1::cudnn_crop(
                                    ni, no, vec3i(insz[0], insz[1], insz[2]),
                                    vec3i(outsz[0], outsz[1], outsz[2])));
    return l;
}

activation activation_type(int i){
    switch (i) {
        case 0: return activation::none;
        case 1: return activation::relu;
        case 2: return activation::logistics;
        case 3: return activation::sigmoid;
        case 4: return activation::tanh;
        case 5: return activation::clipped_relu;
    }
}

znnilayer* conv_layer(int n, int ni, int no, int* insz,
                            int* knsz, DTYPE* kernel, DTYPE* biases, int act){
    znnilayer *l = new znnilayer();
    l->layer = std::unique_ptr<znn::fwd::device::v1::device_layer>(
                                new device::v1::cudnn_conv(n, ni, no,
                                vec3i(insz[0], insz[1], insz[2]),
                                vec3i(knsz[0], knsz[1], knsz[2]),
                                kernel, biases, activation_type(act)));
    return l;
}
