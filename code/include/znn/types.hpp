#pragma once

#include <ostream>
#include <cstdint>
#include <cstddef>
#include <complex>
#include <mutex>
#include <memory>
#include <functional>
#include <zi/vl/vl.hpp>

#include <map>
#include <list>
#include <vector>


namespace znn { namespace fwd {

typedef long double ldouble;

#if defined(ZNN_USE_LONG_DOUBLE_PRECISION)
typedef long double real;
#elif defined(ZNN_USE_DOUBLE_PRECISION)
typedef double real;
#else
typedef float real;
#endif

typedef std::complex<real>   cplx;
typedef std::complex<real>   complex;

typedef int64_t long_t;

typedef zi::vl::vec<long_t,2> vec2i;
typedef zi::vl::vec<long_t,3> vec3i;
typedef zi::vl::vec<long_t,4> vec4i;
typedef zi::vl::vec<long_t,5> vec5i;

typedef std::size_t size_t;

typedef std::lock_guard<std::mutex> guard;

template<typename T>
struct type {};

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Pointer wrappers for type safety

namespace detail {

struct host_pointer_tag  {};
struct device_pointer_tag{};


template<typename T, typename Tag>
class pointer
{
public:
    typedef T   value_type  ;
    typedef T*  pointer_type;

private:
    pointer_type ptr_;

public:
    explicit pointer( pointer_type p = nullptr )
        : ptr_(p)
    {}

    pointer( std::nullptr_t )
        : ptr_(nullptr)
    {}

    pointer( pointer const & other )
             : ptr_(other.get())
    {}

    template<typename O>
    pointer( pointer<O,Tag> const & other,
             typename std::enable_if<std::is_convertible<O,T>::value,
             void*>::type = 0 )
        : ptr_(other.get())
    {}

    pointer& operator=( pointer const & other )
    {
        ptr_ = other.ptr_;
        return *this;
    }

    template<typename O>
    typename std::enable_if<std::is_convertible<O,T>::value,pointer&>::type
    operator=( pointer<O,Tag> const & other )
    {
        ptr_ = other.get();
        return *this;
    }

    pointer_type get() const
    {
        return ptr_;
    }

    operator bool() const
    {
        return ptr_ != nullptr;
    }
};



template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           pointer<T, host_pointer_tag> const & p)
{
    os << "h[" << p.get() << "]";
    return os;
}

template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os,
           pointer<T, device_pointer_tag> const & p)
{
    os << "d[" << p.get() << "]";
    return os;
}

} //  Namespace detail

template<typename T>
using host_ptr = detail::pointer<T,detail::host_pointer_tag>;

template<typename T>
using device_ptr = detail::pointer<T,detail::device_pointer_tag>;

}} // namespace znn::fwd
