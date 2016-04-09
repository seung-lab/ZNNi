#pragma once

#include "znn/host/v1/mfp.hpp"
#include "znn/host/v1/dp_fft_conv.hpp"
#include "znn/host/v1/fft_conv.hpp"
#include "znn/device/tail/network.hpp"

#include <condition_variable>

namespace znn { namespace fwd { namespace device { namespace fusion {

typedef device::tail::cudnn_conv gemm_it;
typedef device::tail::fft_conv   fft_it ;

template<typename Conv>
class network
{
private:
    typedef Conv tail_type;
    typedef std::function<void(host_tensor<float,5>)> callback_type;

private:
    std::vector<std::unique_ptr<host::v1::host_layer>> stage1_;
    std::unique_ptr<device::tail::network<tail_type>>  stage2_;

    vec5i  in_shape_  ;
    vec5i  out_shape_ ;

    long_t memory_required_ = 0;

    std::mutex mtx;
    std::condition_variable cpu_cv;
    std::condition_variable gpu_cv;
    std::condition_variable desctructor_cv;

    long_t cpu_done = 0;
    long_t gpu_done = 0;

    long_t cpu_finished = 0;

    host_tensor<float,5> handover[2];

    std::thread gpu_thread;

private:
    template<typename Callback>
    void gpu_loop(Callback callback)
    {
        for ( long_t r = 0; ; ++r )
        {
            zi::wall_timer wt1;
            zi::wall_timer wt2;

            {
                std::unique_lock<std::mutex> g(mtx);
                while ( (cpu_done <= r) && !cpu_finished )
                {
                    gpu_cv.wait(g);
                }

                if ( cpu_done <= r )
                {
                    return;
                }
            }

            wt1.reset();

            auto output = stage2_->forward(std::move(handover[r%2]));

            {
                std::unique_lock<std::mutex> g(mtx);
                ++gpu_done;
                cpu_cv.notify_all();
            }

            LOG(gpu_loop) << "Took time: " << wt1.elapsed<double>()
                          << " :: " << wt2.lap<double>();
            callback(std::move(output));
        }
    }

public:
    long_t memory_required() const
    {
        return memory_required_;
    }

    ~network()
    {
        {
            std::unique_lock<std::mutex> g(mtx);
            cpu_finished = 1;
            gpu_cv.notify_all();
        }
        gpu_thread.join();
    }

    template<typename ND, class Callback>
    network( ND const & nd, long_t cutoff, Callback cb )
        : stage1_(cutoff)
        , in_shape_(nd.in_shape())
        , out_shape_(nd.out_shape())
    {
        auto net = nd;

        for ( long_t i = 0; i < cutoff; ++i )
        {
            auto l = net.layers()[0];
            if ( l.type == layer_type::convolutional )
            {
                if ( i == 0 )
                {
                    stage1_[i]
                        = make_unique<host::v1::dp_fft_conv>
                        (l.batch_size,
                         l.num_inputs,
                         l.num_outputs,
                         l.in_image_size,
                         l.k_or_w_size,
                         l.hkernels->data(),
                         l.hbiases->data());
                }
                else
                {
                    stage1_[i]
                        = make_unique<host::v1::fft_conv>
                        (l.batch_size,
                         l.num_inputs,
                         l.num_outputs,
                         l.in_image_size,
                         l.k_or_w_size,
                         l.hkernels->data(),
                         l.hbiases->data());
                }
            }
            else
            {
                stage1_[i]
                    = make_unique<host::v1::mfp>
                    (l.batch_size,
                     l.num_inputs,
                     l.in_image_size,
                     l.k_or_w_size);
            }
            memory_required_ = std::max(memory_required_,
                                       stage1_[i]->working_memory());

            net = net.tail();
        }

        for ( auto & l: net.layers() )
        {
            std::cout << l.batch_size << ' '
                      << l.num_inputs << ' '
                      << l.num_outputs << ' '
                      << l.in_image_size << '\n';
        }

        stage2_ = device::tail::network<tail_type>::get(net);

        if ( !stage2_ )
        {
            throw std::logic_error("no feasable network");
        }

        memory_required_ = memory_required_ +
            stage2_->working_memory();

        gpu_thread = std::thread([=]() {
                this->gpu_loop(cb);
            });
    }

    vec5i const & in_shape() const
    {
        return in_shape_;
    }

    vec5i const & out_shape() const
    {
        return out_shape_;
    }

    void forward( host_tensor<real,5> in )
    {
        zi::wall_timer wt;
        long_t r;

        {
            std::unique_lock<std::mutex> g(mtx);
            while ( gpu_done + 2 <= cpu_done )
            {
                cpu_cv.wait(g);
            }
            r = cpu_done;
        }
        wt.reset();

        for ( auto const & l: stage1_ )
        {
            in = l->forward(std::move(in));
        }

        handover[r%2] = std::move(in);

        LOG(cpu_thread) << wt.elapsed<double>();

        {
            std::unique_lock<std::mutex> g(mtx);
            ++cpu_done;
            gpu_cv.notify_all();
        }
    }


};


}}}} // namespace znn::fwd::device::fusion
