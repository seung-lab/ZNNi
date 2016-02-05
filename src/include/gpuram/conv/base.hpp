#pragma once

namespace znn { namespace fwd { namespace gpu3dram {

class base_conv3d
{
public:
    ~base_conv3d() {}
    virtual void forward( float *, float *, float *, float * ) const = 0;
};

}}} // namespace znn::fwd::gpu3dram
