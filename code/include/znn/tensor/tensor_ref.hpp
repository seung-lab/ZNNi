#pragma once

#include "znn/assert.hpp"
#include "znn/types.hpp"
#include "znn/init.hpp"
#include "znn/tensor/tags.hpp"
#include "znn/tensor/memory.hpp"
#include "znn/tensor/base.hpp"
#include "znn/tensor/sub_tensor.hpp"

#include <type_traits>

namespace znn { namespace fwd {

template<class T, size_t D, class A, class TPtr = T const *>
class const_tensor_ref
    : public detail::tensor::value_accessor<T,D,A>
{
private:
    static_assert(D>0, "0 dimensional tensor not allowed");

private:
    typedef detail::tensor::value_accessor<T,D,A> super_type;

public:
    static const size_t dimensionality = super_type::dimensionality;

    typedef typename super_type::element element;
    typedef typename super_type::value_type value_type;
    typedef typename super_type::const_reference const_reference;
    typedef typename super_type::architecture architecture;

protected:
    zi::vl::vec<long_t,D>   extents_;
    zi::vl::vec<long_t,D+1> strides_;
    TPtr ptr_ = nullptr;

private:
    void compute_strides()
    {
        strides_[D] = 1;
        for ( long_t i = D; i > 0; --i )
        {
            strides_[i-1] = strides_[i] * extents_[i-1];
        }
    }

protected:
    const_tensor_ref() noexcept
        : extents_(0)
        , strides_(0)
        , ptr_(nullptr)
    {}

public:
    const_tensor_ref( const_tensor_ref const & other ) noexcept
        : extents_(other.extents_)
        , strides_(other.strides_)
        , ptr_(other.ptr_)
    {}

    explicit const_tensor_ref( TPtr t, zi::vl::vec<long_t,D> const & e )
        : extents_(e)
        , ptr_(t)
    {
        compute_strides();
    }

    template<typename... Args>
    explicit const_tensor_ref( TPtr t, Args&&... args )
        : extents_( zi::vl::vec<long_t,D>(std::forward<Args>(args)...) )
        , ptr_(t)
    {
        compute_strides();
    }

    const_tensor_ref& operator=( const_tensor_ref const & ) = delete;

    const_reference operator[]( long_t i ) const noexcept
    {
        return super_type::access(znn::fwd::type<const_reference>(), i,
                                  ptr_,
                                  extents_.data(),
                                  strides_.data());
    }

    zi::vl::vec<long_t,D> const & shape_vec() const
    {
        return extents_;
    }

    long_t const * shape() const
    {
        return extents_.data();
    }

    long_t num_elements() const
    {
        return strides_[0];
    }

    TPtr data() const
    {
        return ptr_;
    }

    template<class O>
    O const * data_as() const
    {
        return reinterpret_cast<O const *>(ptr_);
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
        STRONG_ASSERT(n <= strides_[0]);
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
                            TPtr>::type
    begin() const
    {
        return ptr_;
    }

    template<typename X = A>
    typename std::enable_if<std::is_same<X,detail::tensor::host_tag>::value,
                            TPtr>::type
    end() const
    {
        return ptr_ + strides_[0];
    }

    zi::vl::vec<long_t,D> const & size() const
    {
        return extents_;
    }

};

template<class T, size_t D, class A>
class tensor_ref
    : public const_tensor_ref<T,D,A,T*>
{
private:
    typedef const_tensor_ref<T,D,A,T*> super_type;

public:
    static const size_t dimensionality = super_type::dimensionality;

    typedef typename super_type::element element;
    typedef typename super_type::value_type value_type;
    typedef typename super_type::reference reference;
    typedef typename super_type::const_reference const_reference;
    typedef typename super_type::architecture architecture;

protected:
    tensor_ref()
        : super_type()
    {}

public:
    explicit tensor_ref( T * t, zi::vl::vec<long_t,D> const & e )
        : super_type(t,e)
    {}

    template<typename... Args>
    explicit tensor_ref( T * t, Args&&... args )
        : super_type(t, std::forward<Args>(args)...)
    {}

    tensor_ref& operator=( tensor_ref const & other ) noexcept
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
    tensor_ref& operator=( Tensor const & other ) noexcept
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
                                  super_type::extents_.data(),
                                  super_type::strides_.data());
    }

    const_reference operator[]( long_t i ) const noexcept
    {
        return super_type::operator[](i);
    }

    T const * data() const
    {
        return this->ptr_;
    }

    T* data()
    {
        return this->ptr_;
    }

    template<class O>
    O* data_as()
    {
        return reinterpret_cast<O*>(this->ptr_);
    }

    template<typename DevTag>
    void load( T const * d, DevTag const & tag )
    {
        detail::tensor::copy_n(d, this->strides_[0], this->ptr_,
                               tag, architecture());
    }

    template<typename DevTag>
    void load_n( T const * d, size_t n, DevTag const & tag )
    {
        STRONG_ASSERT((long_t)(n) <= this->strides_[0]);
        detail::tensor::copy_n(d, n, this->ptr_, tag, architecture());
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


template<class T, size_t D>
using host_tensor_ref = tensor_ref<T,D,detail::tensor::host_tag>;

template<class T, size_t D>
using device_tensor_ref = tensor_ref<T,D,detail::tensor::device_tag>;

template<class T, size_t D>
using host_const_tensor_ref =
    const_tensor_ref<T,D,detail::tensor::host_tag>;

template<class T, size_t D>
using device_const_tensor_ref =
    const_tensor_ref<T,D,detail::tensor::device_tag>;

template<class T>
using host_array_ref = tensor_ref<T,1,detail::tensor::host_tag>;

template<class T>
using device_array_ref = tensor_ref<T,1,detail::tensor::device_tag>;

template<class T>
using host_const_array_ref =
    const_tensor_ref<T,1,detail::tensor::host_tag>;

template<class T>
using device_const_array_ref =
    const_tensor_ref<T,1,detail::tensor::device_tag>;



}} // namespace znn::fwd
