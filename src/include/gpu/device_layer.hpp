#pragma once

#include "../assert.hpp"
#include "memory.hpp"
#include "handle.hpp"

namespace znn { namespace fwd {

class device_layer
{
public:
    typedef device_layer        layer_type;
    typedef device_array<float> array_type;
    typedef gpu::handle_t       handle_type;

    virtual ~device_layer() {}
    virtual device_array<float> forward( device_array<float> ) const = 0;

    virtual long_t permanent_memory_required() const
    {
        UNIMPLEMENTED();
    }

    virtual long_t working_memory_required() const
    {
        UNIMPLEMENTED();
    }

};


}} // namespace znn::fwd
