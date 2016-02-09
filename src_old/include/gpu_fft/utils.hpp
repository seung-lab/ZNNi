#include <cufft.h>
#include <cstddef>

#include "../types.hpp"

namespace znn { namespace fwd {

void div_all_by( float*, float*, float ) noexcept;
void add_to( cuComplex*, cuComplex*, cuComplex*, float) noexcept;
void mul_add( cuComplex*, cuComplex*, cuComplex*, cuComplex* ) noexcept;

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


}} // namespace znn::fwd
