#pragma once

#include "../../types.hpp"
#include "../../assert.hpp"
#include "../../memory.hpp"
#include "../../layer.hpp"

namespace znn { namespace fwd { namespace tbb {

using cpu_convolutional_layer_base =
    ::znn::fwd::cpu::cpu_convolutional_layer_base;

}}} // namespace znn::fwd::tbb
