#pragma once

#include "znn/types.hpp"
#include "znn/tensor/tags.hpp"
#include "znn/tensor/device_reference.hpp"

namespace znn { namespace fwd { namespace detail { namespace tensor {

template<typename, size_t, typename>
class tensor;

template<typename, size_t, typename>
class sub_tensor;

template<typename, size_t, typename>
class const_sub_tensor;

template<typename T, size_t NumDims, typename Arch>
class value_accessor
{
public:
    static const size_t dimensionality = NumDims;

    typedef T                                element        ;
    typedef tensor<T,NumDims-1,Arch>         value_type     ;
    typedef sub_tensor<T,NumDims-1,Arch>     reference      ;
    typedef const_sub_tensor<T,NumDims,Arch> const_reference;
    typedef Arch                             architecture   ;

protected:
    template<typename Reference>
    Reference access(znn::fwd::type<Reference>,
                     long_t index, T* base,
                     long_t const * extents,
                     long_t const * strides) const noexcept
    {
        ZI_ASSERT(index<extents[0]);
        return Reference(base + index * strides[1],extents + 1, strides + 1);
    }
};

template<typename T>
class value_accessor<T,1,host_tag>
{
public:
    static const size_t dimensionality = 1;

    typedef T         element        ;
    typedef T         value_type     ;
    typedef T &       reference      ;
    typedef T const & const_reference;
    typedef host_tag  architecture   ;

protected:
    template<typename Reference>
    Reference access(znn::fwd::type<Reference>,
                     long_t index, T* base,
                     long_t const *,
                     long_t const * strides) const noexcept
    {
        //ZI_ASSERT(index<extents[0]);
        return *(base + index * strides[1]);
    }
};


template<typename T>
class value_accessor<T,1,device_tag>
{
public:
    static const size_t dimensionality = 1;

    typedef T                         element        ;
    typedef T                         value_type     ;
    typedef device_reference<T>       reference      ;
    typedef const_device_reference<T> const_reference;
    typedef device_tag                architecture   ;

protected:
    template<typename Reference>
    Reference access(znn::fwd::type<Reference>,
                     long_t index, T* base,
                     long_t const *,
                     long_t const * strides) const noexcept
    {
        //ZI_ASSERT(index<extents[0]);
        return Reference(base + index * strides[1]);
    }
};


}}}} // namespace znn::fwd::detail::tensor
