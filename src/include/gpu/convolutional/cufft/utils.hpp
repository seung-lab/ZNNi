#pragma once

#include <cufft.h>

#include "../../../types.hpp"

namespace znn { namespace fwd {

void div_all_by( float*, float*, float ) noexcept;
void add_to( cuComplex*, cuComplex*, cuComplex*, float) noexcept;
void add_to( float*, float*, float*, float) noexcept;
void mul_add( cuComplex*, cuComplex*, cuComplex*, cuComplex* ) noexcept;
void mul_add2( cuComplex*, cuComplex*, cuComplex*, int ) noexcept;

void stage_1_scatter( int, int, float*, float*, long_t ) noexcept;
void stage_1_gather( int, int, float*, float*, long_t ) noexcept;

void stage_2_scatter( int, int, int, cuComplex*, cuComplex*, long_t ) noexcept;
void stage_2_gather( int, int, int, cuComplex*, cuComplex*, long_t ) noexcept;

class kernel_exploder
{
private:
    int*   workspace;
    size_t len;
    size_t olen;

public:
    kernel_exploder( int*, vec3i const &, vec3i const &, size_t );
    void explode( float*, float* );
};


class image_imploder
{
private:
    int*   workspace;
    size_t len;

public:
    image_imploder( int*, vec3i const &, vec3i const & );
    void implode( float*, float* );
};

class image_scatter
{
private:
    int*   workspace;
    size_t len;
    size_t olen;

public:
    image_scatter( int *, vec3i const &, vec3i const & );
    void scatter( float*, float* );
};


class image_gather
{
private:
    int*   workspace;
    size_t len;

public:
    image_gather( int *, vec3i const &, vec3i const &, vec3i const & );
    void gather( float*, float* );
};


}} // namespace znn::fwd
