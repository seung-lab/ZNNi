#pragma once

#include "znn/types.hpp"

//#define _GNU_SOURCE
#include <sched.h>
#include <atomic>

namespace znn { namespace fwd { namespace host {

class thread_distributor
{
private:
    int  chips;
    int  cores_per_chip;
    bool hyperthreading;

    std::atomic<int> counter;
    int total;

public:
    thread_distributor(int a = 4, int b = 36, bool c = true)
        : chips(a)
        , cores_per_chip(b)
        , hyperthreading(c)
        , counter(0)
        , total(a*b)
    {}

    thread_distributor( thread_distributor const & ) = delete;
    thread_distributor operator=( thread_distributor const & ) = delete;

    int next()
    {
        int r = counter++;

        if ( hyperthreading )
        {
            r = r % total;
            if ( r < (total / 2))
            {
                return r * 2 + 1;
            }
            else
            {
                r -= (total / 2);
                int chip    = r % 4;
                int on_chip = r / 4;

                return chip * cores_per_chip + on_chip * 2;
            }
        }
        else
        {
            return r % total;
        }
    }

};

class thread_pin
{
private:
    cpu_set_t old_set;

public:
    explicit thread_pin( thread_distributor & td )
    {
        int cpu = td.next();
        sched_getaffinity(0, sizeof(old_set), &old_set);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET( cpu , &cpuset);

        sched_setaffinity(0, sizeof(cpuset), &cpuset);
    }

    ~thread_pin()
    {
        sched_setaffinity(0, sizeof(old_set), &old_set);
    }

    thread_pin( thread_pin const & ) = delete;
    thread_pin& operator=( thread_pin const & ) = delete;

};


}}} // namespace znn::fwd::host
