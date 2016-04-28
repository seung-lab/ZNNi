#pragma once

#if defined(ZNN_USE_MKL_FFT)
#  include "znn/host/common/fft2d/fftmkl.hpp"
#else
#  include "znn/host/common/fft2d/fftw.hpp"
#endif

#include <map>
#include <mutex>
#include <zi/utility/singleton.hpp>

namespace znn { namespace fwd { namespace host {

class padded_pruned_fft2d_plans_impl
{
private:
    std::mutex                                                           m_  ;
    std::map<vec2i, std::map<vec2i, padded_pruned_fft2d_transformer*>>   map_;

public:
    ~padded_pruned_fft2d_plans_impl()
    {
        for ( auto & p: map_ )
            for ( auto & q: p.second )
                delete q.second;
    }

    padded_pruned_fft2d_plans_impl(): m_(), map_()
    {
    }

    padded_pruned_fft2d_transformer* get( vec2i const & r, vec2i const & c )
    {
        typedef  padded_pruned_fft2d_transformer* ret_type;

        guard g(m_);

        ret_type& ret = map_[r][c];
        if ( ret ) return ret;

        ret = new padded_pruned_fft2d_transformer(r,c);
        return ret;
    }

}; // class padded_pruned_fft2d_plans_impl


namespace {
padded_pruned_fft2d_plans_impl& padded_pruned_fft2d_plans =
    zi::singleton<padded_pruned_fft2d_plans_impl>::instance();
} // anonymous namespace

}}} // namespace znn::fwd::host
