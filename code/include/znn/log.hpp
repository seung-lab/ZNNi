#pragma once

#include "znn/types.hpp"

#include <zi/utility/singleton.hpp>
#include <zi/utility/non_copyable.hpp>
#include <zi/time.hpp>

#include <iostream>
#include <string>

namespace znn { namespace fwd {

namespace detail {

class ostream_wrapper: zi::non_copyable
{
public:
    virtual ~ostream_wrapper() {};
    virtual void print(std::string const &) = 0;
};

template<typename T>
class ostream_wrapper_tpl: public ostream_wrapper
{
private:
    T& stream_;

public:
    explicit ostream_wrapper_tpl( T & s )
        : stream_(s)
    {}

    void print(std::string const & d) override
    {
        stream_ << d << std::endl;
    }
};

template<typename T>
std::unique_ptr<ostream_wrapper_tpl<T>>
get_ostream_wrapper( T & s )
{
    return make_unique<ostream_wrapper_tpl<T>>(s);
}

class logger_impl: zi::non_copyable
{
private:
    std::mutex                       mutex_ ;
    std::unique_ptr<ostream_wrapper> stream_;

public:
    logger_impl()
        : mutex_()
        , stream_(get_ostream_wrapper(std::cout))
    {
    }

    template<typename T>
    void set_ostream( T & os )
    {
        guard g(mutex_);
        stream_ = get_ostream_wrapper(os);
    }

    void print(std::string const & s)
    {
        stream_->print(s);
    }

}; // class log_output_impl

namespace {
logger_impl& logger = zi::singleton<logger_impl>::instance();
} // namespace

class log_token: zi::non_copyable
{
public:
    std::ostringstream i_;

    ~log_token()
    {
        logger.print(i_.str());
    }

    template< typename T >
    log_token& operator<< ( T const & v )
    {
        i_ << v;
        return *this;
    }

}; // log_token

} // namespace detail

using detail::logger;

}} // namespace znn::fwd


#define LOG(what) (::znn::fwd::detail::log_token())     \
    << "### LOG[" << #what << "] :: "
