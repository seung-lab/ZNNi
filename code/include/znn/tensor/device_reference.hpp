#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tags.hpp"
#include "znn/tensor/memory.hpp"

namespace znn { namespace fwd { namespace detail { namespace tensor {

template<typename T>
class const_device_reference
{
protected:
    T* p;

public:
    explicit const_device_reference(T* t) noexcept
        : p(t)
    {}

    const_device_reference( const_device_reference const & other ) noexcept
        : p(other.p)
    {}

    operator T() const noexcept
    {
        return detail::tensor::load(p, device_tag());
    }

    const_device_reference& operator=(const_device_reference const &) = delete;
};

template<typename T>
class device_reference
    : public const_device_reference<T>
{
private:
    typedef const_device_reference<T> super_type;

public:
    explicit device_reference(T* t) noexcept
        : super_type(t)
    {}

    device_reference(device_reference const & other) noexcept
        : super_type(other.p)
    {}

    device_reference& operator=(device_reference const & other) noexcept
    {
        T x = detail::tensor::load(other.p, device_tag());
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }

    template<typename O>
    device_reference& operator=(O const & other) noexcept
    {
        detail::tensor::store(super_type::p, static_cast<T>(other),
                              device_tag());
        return *this;
    }

    template<typename O>
    device_reference& operator+=(O const & other) noexcept
    {
        T x = detail::tensor::load(super_type::p, device_tag());
        x += static_cast<T>(other);
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }

    template<typename O>
    device_reference& operator-=(O const & other) noexcept
    {
        T x = detail::tensor::load(super_type::p, device_tag());
        x -= static_cast<T>(other);
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }

    template<typename O>
    device_reference& operator*=(O const & other) noexcept
    {
        T x = detail::tensor::load(super_type::p, device_tag());
        x *= static_cast<T>(other);
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }

    template<typename O>
    device_reference& operator/=(O const & other) noexcept
    {
        T x = detail::tensor::load(super_type::p, device_tag());
        x /= static_cast<T>(other);
        detail::tensor::store(super_type::p, x, device_tag());
        return *this;
    }

};

}}}} // namespace znn::fwd::detail::tensor
