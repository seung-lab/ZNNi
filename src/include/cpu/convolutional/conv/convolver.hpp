#pragma once

#if defined(ZNN_USE_MKL_CONVOLUTION)
#  include "mkl.hpp"
#else
#  include "naive.hpp"
#endif
