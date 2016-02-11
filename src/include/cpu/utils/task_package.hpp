#pragma once

#include "global_task_manager.hpp"

#include "../../memory.hpp"
#include "../../types.hpp"

#include <vector>
#include <thread>
#include <algorithm>
#include <utility>
#include <functional>
#include <condition_variable>


namespace znn { namespace fwd {

class task_package
{
public:
    task_package(const task_package&) = delete;
    task_package& operator=(const task_package&) = delete;

    task_package(task_package&& other) = delete;
    task_package& operator=(task_package&&) = delete;


private:
    typedef std::function<void(void*)> callable_t;

private:
    std::vector<callable_t>       tasks_;
    std::atomic<int>              size_ ;
    std::atomic<int>              running_threads_;

    std::mutex                    m_ ;
    std::condition_variable       cv_;

    std::size_t                   threads_;

private:
    void loop( void * stack )
    {
        while (1)
        {
            int tid = --size_;
            if ( tid >= 0 )
            {
                tasks_[tid](stack);
            }
            else
            {
                if ( --running_threads_ == 0 )
                {
                    std::lock_guard<std::mutex> g(m_);
                    cv_.notify_one();
                }
                return;
            }
        }
    }

public:
    task_package( std::size_t n = 1000000,
                  std::size_t t = std::thread::hardware_concurrency() )
        : tasks_(n)
        , size_(0)
        , running_threads_(0)
        , m_()
        , cv_()
        , threads_(t)
    { }

    template<typename... Args>
    void add_task(Args&&... args)
    {
        tasks_[size_++] = std::bind(std::forward<Args>(args)...,
                                    std::placeholders::_1);
    }

    long_t concurrency() const
    {
        return static_cast<long_t>(threads_);
    }

    void execute( std::size_t stack_size = 0 )
    {
        std::size_t n_workers = size_.load();

        while ( n_workers > 3 * threads_ )
        {
            n_workers /= 2;
        }

        if ( n_workers == 0 && size_.load() > 0 )
        {
            n_workers = 1;
        }

        if ( n_workers > 0 )
        {
            host_array<char> stack = get_array<char>(n_workers*stack_size);

            running_threads_ = static_cast<int>(n_workers);

            for ( std::size_t i = 1; i < n_workers; ++i )
            {
                global_task_manager.schedule(&task_package::loop, this,
                                             stack.get() + i * stack_size );
            }

            // use this thread as well, that guarantees execution
            // completion as there will always be at least one
            // resource [thread] available to complete the tasks on
            // the queue
            loop(stack.get());

            {
                std::unique_lock<std::mutex> g(m_);
                while ( running_threads_.load() > 0 )
                {
                    cv_.wait(g);
                }
            }
            size_ = 0;
        }
    }

};

}} // namespace znn::fwd
