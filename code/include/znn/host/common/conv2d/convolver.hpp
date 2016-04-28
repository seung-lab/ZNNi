#pragma once

#if defined(ZNN_USE_MKL_CONVOLUTION)
#  include "znn/host/common/conv2d/mkl.hpp"
#else
#  include "znn/host/common/conv2d/naive.hpp"
#endif
