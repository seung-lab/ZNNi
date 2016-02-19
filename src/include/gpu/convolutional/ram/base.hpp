#pragma once

#include "../../../types.hpp"
#include "../../../layer.hpp"

namespace znn { namespace fwd { namespace gpu {

class gpuram_layer_base
    : public convolutional_layer_base
{
public:
    gpuram_layer_base( long_t n, long_t fin, long_t fout,
                       vec3i const & is, vec3i const & ks )
        : convolutional_layer_base(n,fin,fout,is,ks)
    {}

    virtual long_t workspace_size() const = 0;
    virtual void forward( float*, float*, float*, float*, float, float* ) const = 0;

};

}}} // namespace znn::fwd::gpu
