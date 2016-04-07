#pragma once

#ifdef __unix__

#ifndef ZNNI_NUM_CHIPS
#  define ZNNI_NUM_CHIPS 4
#endif

#ifndef ZNNI_CORES_PER_CHIP
#  define ZNNI_CORES_PER_CHIP 36
#endif

#ifndef ZNNI_HYPERTHREADING
#  define ZNNI_HYPERTHREADING 1
#endif

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
    thread_distributor(int a = ZNNI_NUM_CHIPS, int b = ZNNI_CORES_PER_CHIP,
                       bool c = ZNNI_HYPERTHREADING)
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
                int chip    = r % chips;
                int on_chip = r / chips;

                return chip * cores_per_chip + on_chip * 2 + 1;
                //return r * 2 + 1;
            }
            else
            {
                r -= (total / 2);
                int chip    = r % chips;
                int on_chip = r / chips;

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
    int cpu;

public:
    explicit thread_pin( thread_distributor & td )
    {
        cpu = td.next();
        sched_getaffinity(0, sizeof(old_set), &old_set);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET( cpu , &cpuset);

        sched_setaffinity(0, sizeof(cpuset), &cpuset);
    }

    int location() const
    {
        return cpu;
    }

    ~thread_pin()
    {
        sched_setaffinity(0, sizeof(old_set), &old_set);
    }

    thread_pin( thread_pin const & ) = delete;
    thread_pin& operator=( thread_pin const & ) = delete;

};

class cpu_pin
{
private:
    cpu_set_t old_set;

public:
    explicit cpu_pin( int vcore )
    {
        sched_getaffinity(0, sizeof(old_set), &old_set);

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        vcore /= ZNNI_CORES_PER_CHIP;
        vcore *= ZNNI_CORES_PER_CHIP;
        for ( int i = 0; i < ZNNI_CORES_PER_CHIP; ++i )
        {
            CPU_SET(vcore+i, &cpuset);
        }

        sched_setaffinity(0, sizeof(cpuset), &cpuset);
    }

    ~cpu_pin()
    {
        sched_setaffinity(0, sizeof(old_set), &old_set);
    }

    cpu_pin( thread_pin const & ) = delete;
    cpu_pin& operator=( thread_pin const & ) = delete;

};


}}} // namespace znn::fwd::host

#else

namespace znn { namespace fwd { namespace host {

class thread_distributor
{
public:
    explicit thread_distributor(int=0,int=0,bool=true)
    {}

    thread_distributor( thread_distributor const & ) = delete;
    thread_distributor operator=( thread_distributor const & ) = delete;

};

class thread_pin
{
public:
    explicit thread_pin( thread_distributor & )
    {}

    thread_pin( thread_pin const & ) = delete;
    thread_pin& operator=( thread_pin const & ) = delete;

};



}}} // namespace znn::fwd::host

#endif
