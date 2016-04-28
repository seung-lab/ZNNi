#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"

#include "znn/device/common/utils.hpp"

#include <stdexcept>
#include <cudnn.h>


namespace znn { namespace fwd { namespace device { namespace cudnn {

class tensor_descriptor2d
{
private:
    cudnnTensorDescriptor_t handle_;

public:
    tensor_descriptor2d()
    {
        tryCUDNN(cudnnCreateTensorDescriptor(&handle_));
    }

    ~tensor_descriptor2d()
    {
        checkCUDNN(cudnnDestroyTensorDescriptor(handle_));
    }

    cudnnTensorDescriptor_t const & handle() const
    {
        return handle_;
    }

    void set( int n, int c, int h, int w,
              int N, int C, int H, int W )
    {
        int dims[4] = {n,c,h,w};
        int stri[4] = {N,C,H,W};
        tryCUDNN(cudnnSetTensorNdDescriptor(handle_,
                                            CUDNN_DATA_FLOAT,
                                            4, dims, stri));
    }

    void set( int n, int c, int h, int w )
    {
        set(n,c,h,w,c*h*w,h*w,w,1);
    }

    tensor_descriptor2d(tensor_descriptor2d const &) = delete;
    tensor_descriptor2d& operator=(tensor_descriptor2d const &) = delete;

};


class kernel_descriptor2d
{
private:
    cudnnFilterDescriptor_t handle_;

public:
    kernel_descriptor2d()
    {
        tryCUDNN(cudnnCreateFilterDescriptor(&handle_));
    }

    ~kernel_descriptor2d()
    {
        checkCUDNN(cudnnDestroyFilterDescriptor(handle_));
    }

    cudnnFilterDescriptor_t const & handle() const
    {
        return handle_;
    }

    void set( int n, int c, int h, int w )
    {
        int dims[4] = {n,c,h,w};
        tryCUDNN(cudnnSetFilterNdDescriptor(handle_,
                                            CUDNN_DATA_FLOAT,
                                            4, dims));

    }

    kernel_descriptor2d(kernel_descriptor2d const &) = delete;
    kernel_descriptor2d& operator=(kernel_descriptor2d const &) = delete;

};



class convolution_descriptor2d
{
private:
    cudnnConvolutionDescriptor_t handle_;

public:
    convolution_descriptor2d()
    {
        tryCUDNN(cudnnCreateConvolutionDescriptor(&handle_));
    }

    ~convolution_descriptor2d()
    {
        checkCUDNN(cudnnDestroyConvolutionDescriptor(handle_));
    }

    cudnnConvolutionDescriptor_t const & handle() const
    {
        return handle_;
    }

    void set( int px = 0, int py = 0 )
    {
        int pad[2] = {px,py};
        int ones[2] = {1,1};

        tryCUDNN( cudnnSetConvolutionNdDescriptor(
                      handle_,
                      2, pad, ones, ones,
                      CUDNN_CONVOLUTION,
                      //CUDNN_CROSS_CORRELATION,
                      CUDNN_DATA_FLOAT) );

    }

    convolution_descriptor2d(convolution_descriptor2d const &) = delete;
    convolution_descriptor2d& operator=(convolution_descriptor2d const &) = delete;

};




class pooling_descriptor2d
{
private:
    cudnnPoolingDescriptor_t handle_;

public:
    pooling_descriptor2d()
    {
        tryCUDNN(cudnnCreatePoolingDescriptor(&handle_));
    }

    ~pooling_descriptor2d()
    {
        checkCUDNN(cudnnDestroyPoolingDescriptor(handle_));
    }

    cudnnPoolingDescriptor_t const & handle() const
    {
        return handle_;
    }

    void set( int px = 0, int py = 0 )
    {
        int window[2]  = { px, py };
        int padding[2] = {0,0};

        tryCUDNN( cudnnSetPoolingNdDescriptor(
                      handle_,
                      CUDNN_POOLING_MAX,
                      2, window, padding, window ));
    }

    pooling_descriptor2d(pooling_descriptor2d const &) = delete;
    pooling_descriptor2d& operator=(pooling_descriptor2d const &) = delete;

};


}}}} // namespace znn::fwd::device::cudnn
