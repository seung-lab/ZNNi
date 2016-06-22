#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tensor.hpp"
#include "znn/host/v1/host_layer.hpp"

namespace znn { namespace fwd { namespace host { namespace v1 {

class crop
    : public crop_layer<host_layer>
{
private:
    vec3i c_; // crop size

public:
    long_t resident_memory() const override
    {
        return 0;
    }

    long_t working_memory() const override
    {
        return input_memory + output_memory;
    }

    host_tensor<float,5> forward( host_tensor<float,5> in ) const override
    {
        host_tensor<real,5> out(output_shape);

        for (long_t b=0; b<output_shape[0]; b++)
            for (long_t i=0; i<output_shape[1]; i++)
                for (long_t x=0; x<output_shape[2]; x++)
                    for (long_t y=0; y<output_shape[3]; y++)
                        for (long_t z=0; z<output_shape[4]; z++)
                            out[b][i][x][y][z] = in[b][i][x+c_[0]][y+c_[1]][z+c_[2]];
        return out;
    }

    crop( long_t n, long_t fin,
                vec3i const & is, vec3i const & c )
        : crop_layer<host_layer>(n,fin,is,c)
    {
        c_ = c;
    }
};


}}}} // namespace znn::fwd::host::v1
