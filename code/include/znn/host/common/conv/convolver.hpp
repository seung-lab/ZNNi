#pragma once

#if defined(ZNN_USE_MKL_CONVOLUTION)
#  include "znn/host/common/conv/mkl.hpp"
#else
#  include "znn/host/common/conv/naive.hpp"
#endif
