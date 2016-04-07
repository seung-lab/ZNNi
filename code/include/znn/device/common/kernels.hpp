#pragma once

#include "znn/types.hpp"

namespace znn { namespace fwd { namespace device {

void div_all_by( float*, float*, float ) noexcept;
void add_to( float*, float*, float*, float) noexcept;
void mul_add( cuComplex*, cuComplex*, cuComplex*, cuComplex* ) noexcept;

void stage_1_scatter( int, int, float const*, float*, long_t ) noexcept;
void stage_1_gather( int, int, float*, float const*, long_t ) noexcept;

void stage_2_scatter( int, int, int, cuComplex const*, cuComplex*, long_t ) noexcept;
void stage_2_gather( int, int, int, cuComplex*, cuComplex const*, long_t ) noexcept;


}}} // namespace znn::fwd::device
