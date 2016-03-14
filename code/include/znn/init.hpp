#pragma once

#include "znn/types.hpp"
#include <random>
#include <mutex>

namespace znn { namespace fwd {

template<typename T>
void random_initialize(T* ptr, size_t n, T d = static_cast<T>(0.1)) noexcept
{
    static std::mt19937 rng = std::mt19937(1234);
    static std::mutex   m;

    std::uniform_real_distribution<T> dis(-d,d);

    {
        guard g(m);

        for ( size_t i = 0; i < n; ++i )
        {
            ptr[i] = dis(rng);
        }
    }
}

}} // namespace znn::fwd
