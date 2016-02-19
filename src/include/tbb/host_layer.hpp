#pragma once


#include "../types.hpp"
#include "../memory.hpp"
#include "../assert.hpp"

namespace znn { namespace fwd { namespace tbb {

class host_layer
{
public:
    typedef host_layer       layer_type;
    typedef host_array<real> array_type;

    virtual ~host_layer() {}
    virtual host_array<real> forward( host_array<real> ) const = 0;
};


}}} // namespace znn::fwd::tbb
