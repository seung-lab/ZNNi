#pragma once

#include "znn/assert.hpp"
#include "znn/types.hpp"
#include "znn/tensor/tags.hpp"
#include "znn/tensor/memory.hpp"
#include "znn/tensor/base.hpp"

#include <algorithm>

namespace znn { namespace fwd { namespace detail { namespace tensor {



template<class T, size_t D, class A>
class const_sub_tensor
    : public value_accessor<T,D,A>
{
private:
    static_assert(D>0, "0 dimensional tensor not allowed");

private:
    typedef value_accessor<T,D,A> super_type;

public:
    static const size_t dimensionality = super_type::dimensionality;

    typedef typename super_type::element element;
    typedef typename super_type::value_type value_type;
    typedef typename super_type::const_reference const_reference;
    typedef typename super_type::architecture architecture;

protected:
    friend class value_accessor<T,D+1,A>;

protected:
    T*             ptr_     ;
    long_t const * extents_ ;
    long_t const * strides_ ;

protected:
    const_sub_tensor( T* p, long_t const * s, long_t const * e) noexcept
        : ptr_(p), extents_(s), strides_(e)
    {}

public:
    const_sub_tensor( const_sub_tensor const & other ) noexcept
        : ptr_(other.ptr_), extents_(other.extents_), strides_(other.strides_)
    {}

    const_sub_tensor& operator=( const_sub_tensor const & ) = delete;

    const_reference operator[]( long_t i ) const noexcept
    {
        return super_type::access(znn::fwd::type<const_reference>(),
                                  i, ptr_, extents_, strides_);
    }

    zi::vl::vec<long_t,D> shape_vec() const
    {
        return zi::vl::vec<long_t,D>(zi::vl::load, extents_);
    }

    long_t const * shape() const
    {
        return extents_;
    }

    long_t num_elements() const
    {
        return strides_[0];
    }

    T const * data() const
    {
        return ptr_;
    }

    template<class O>
    O const * data_as() const
    {
        return reinterpret_cast<O const*>(ptr_);
    }

    template<typename DevTag>
    void store( T * d, DevTag const & tag ) const
    {
        detail::tensor::copy_n(ptr_, strides_[0], d,
                               architecture(), tag);
    }

    template<typename DevTag>
    void store_n( T * d, size_t n, DevTag const & tag ) const
    {
        STRONG_ASSERT(n<=strides_[0]);
        detail::tensor::copy_n(ptr_, n, d, architecture(), tag);
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::host_tag>::value,
                            host_ptr<T const> >::type
    ptr() const
    {
        return host_ptr<T const>(ptr_);
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::device_tag>::value,
                            device_ptr<T const> >::type
    ptr() const
    {
        return device_ptr<T const>(ptr_);
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::host_tag>::value,
                            T const*>::type
    begin() const
    {
        return ptr_;
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::host_tag>::value,
                            T const*>::type
    end() const
    {
        return ptr_ + strides_[0];
    }

};


template<class T, size_t D, class A>
class sub_tensor
    : public const_sub_tensor<T,D,A>
{
private:
    typedef const_sub_tensor<T,D,A> super_type;

public:
    static const size_t dimensionality = super_type::dimensionality;

    typedef typename super_type::element element;
    typedef typename super_type::value_type value_type;
    typedef typename super_type::reference reference;
    typedef typename super_type::const_reference const_reference;
    typedef typename super_type::architecture architecture;

protected:
    friend class value_accessor<T,D+1,A>;

protected:
    sub_tensor( T* p, long_t const * s, long_t const * e) noexcept
        : super_type(p,s,e)
    {}

public:
    sub_tensor& operator=(sub_tensor const & other)
    {
        if ( this != &other )
        {
            ZI_ASSERT(std::equal(this->shape(),
                                 this->shape()+dimensionality,other.shape()));
            detail::tensor::copy_n(other.data(),this->num_elements(),data(),
                                   architecture(), architecture());
        }
        return *this;
    }

    template<typename Tensor>
    sub_tensor& operator=( Tensor const & other ) noexcept
    {
        static_assert(dimensionality==Tensor::dimensionality,
                      "= enabled for tensors with same dimensionality");

        typedef typename Tensor::architecture other_architecture;

        ZI_ASSERT(std::equal(this->shape(),
                             this->shape()+dimensionality,other.shape()));
        detail::tensor::copy_n(other.data(),this->num_elements(),data(),
                               other_architecture(), architecture());
        return *this;
    }

    reference operator[]( long_t i ) noexcept
    {
        return super_type::access(znn::fwd::type<reference>(), i,
                                  super_type::ptr_,
                                  super_type::extents_,
                                  super_type::strides_);
    }

    T const * data() const
    {
        return super_type::ptr_;
    }

    T* data()
    {
        return super_type::ptr_;
    }

    template<class O>
    O* data_as()
    {
        return reinterpret_cast<O*>(super_type::ptr_);
    }

    template<typename DevTag>
    void load( T const * d, DevTag const & tag )
    {
        detail::tensor::copy_n(d, this->num_elements(), super_type::ptr_,
                               tag, architecture());
    }

    template<typename DevTag>
    void load_n( T const * d, size_t n, DevTag const & tag )
    {
        STRONG_ASSERT(n<=this->num_elements());
        detail::tensor::copy_n(d, n, super_type::ptr_,
                               tag, architecture());
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::host_tag>::value,
                            host_ptr<T> >::type
    ptr()
    {
        return host_ptr<T>(this->ptr_);
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::device_tag>::value,
                            device_ptr<T> >::type
    ptr()
    {
        return device_ptr<T>(this->ptr_);
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::host_tag>::value,
                            host_ptr<T const> >::type
    ptr() const
    {
        return host_ptr<T const>(this->ptr_);
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::device_tag>::value,
                            device_ptr<T const> >::type
    ptr() const
    {
        return device_ptr<T const>(this->ptr_);
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::host_tag>::value,
                            T*>::type
    begin()
    {
        return this->ptr_;
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::host_tag>::value,
                            T*>::type
    end()
    {
        return this->ptr_ + this->strides_[0];
    }

};

}}}} // namespace znn::fwd::detail::tensor
