#pragma once

#include <cudnn.h>

#include "cuda_utils.hpp"
#include "../types.hpp"


namespace znn { namespace fwd { namespace gpu3dram {

class conv_base
{
public:
    virtual ~conv_base() {}

    virtual long_t workspace_memory() const = 0;
    virtual long_t memory() const = 0;

    virtual void forward(float*, float*, float*, float *) const = 0;
};

}}} // namespace znn::fwd::gpu3dram
