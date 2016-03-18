#pragma once

#include "znn/types.hpp"
#include "znn/assert.hpp"

#include "znn/device/common/utils.hpp"

#include <stdexcept>
#include <cudnn.h>


namespace znn { namespace fwd { namespace device { namespace cudnn {

class tensor_descriptor
{
private:
    cudnnTensorDescriptor_t handle_;

public:
    tensor_descriptor()
    {
        tryCUDNN(cudnnCreateTensorDescriptor(&handle_));
    }

    ~tensor_descriptor()
    {
        checkCUDNN(cudnnDestroyTensorDescriptor(handle_));
    }

    cudnnTensorDescriptor_t const & handle() const
    {
        return handle_;
    }

    void set( int n, int c, int d, int h, int w,
              int N, int C, int D, int H, int W )
    {
        int dims[5] = {n,c,d,h,w};
        int stri[5] = {N,C,D,H,W};
        tryCUDNN(cudnnSetTensorNdDescriptor(handle_,
                                            CUDNN_DATA_FLOAT,
                                            5, dims, stri));
    }

    void set( int n, int c, int d, int h, int w )
    {
        set(n,c,d,h,w,c*d*h*w,d*h*w,h*w,w,1);
    }

    tensor_descriptor(tensor_descriptor const &) = delete;
    tensor_descriptor& operator=(tensor_descriptor const &) = delete;

};


class kernel_descriptor
{
private:
    cudnnFilterDescriptor_t handle_;

public:
    kernel_descriptor()
    {
        tryCUDNN(cudnnCreateFilterDescriptor(&handle_));
    }

    ~kernel_descriptor()
    {
        checkCUDNN(cudnnDestroyFilterDescriptor(handle_));
    }

    cudnnFilterDescriptor_t const & handle() const
    {
        return handle_;
    }

    void set( int n, int c, int d, int h, int w )
    {
        int dims[5] = {n,c,d,h,w};
        tryCUDNN(cudnnSetFilterNdDescriptor(handle_,
                                            CUDNN_DATA_FLOAT,
                                            5, dims));

    }

    kernel_descriptor(kernel_descriptor const &) = delete;
    kernel_descriptor& operator=(kernel_descriptor const &) = delete;

};



class convolution_descriptor
{
private:
    cudnnConvolutionDescriptor_t handle_;

public:
    convolution_descriptor()
    {
        tryCUDNN(cudnnCreateConvolutionDescriptor(&handle_));
    }

    ~convolution_descriptor()
    {
        checkCUDNN(cudnnDestroyConvolutionDescriptor(handle_));
    }

    cudnnConvolutionDescriptor_t const & handle() const
    {
        return handle_;
    }

    void set( int px = 0, int py = 0, int pz = 0 )
    {
        int pad[3] = {px,py,pz};
        int ones[3] = {1,1,1};

        tryCUDNN( cudnnSetConvolutionNdDescriptor(
                      handle_,
                      3, pad, ones, ones,
                      CUDNN_CONVOLUTION,
                      //CUDNN_CROSS_CORRELATION,
                      CUDNN_DATA_FLOAT) );

    }

    convolution_descriptor(convolution_descriptor const &) = delete;
    convolution_descriptor& operator=(convolution_descriptor const &) = delete;

};




class pooling_descriptor
{
private:
    cudnnPoolingDescriptor_t handle_;

public:
    pooling_descriptor()
    {
        tryCUDNN(cudnnCreatePoolingDescriptor(&handle_));
    }

    ~pooling_descriptor()
    {
        checkCUDNN(cudnnDestroyPoolingDescriptor(handle_));
    }

    cudnnPoolingDescriptor_t const & handle() const
    {
        return handle_;
    }

    void set( int px = 0, int py = 0, int pz = 0 )
    {
        int window[3]  = { px, py, pz };
        int padding[3] = {0,0,0};

        tryCUDNN( cudnnSetPoolingNdDescriptor(
                      handle_,
                      CUDNN_POOLING_MAX,
                      3, window, padding, window ));
    }

    pooling_descriptor(pooling_descriptor const &) = delete;
    pooling_descriptor& operator=(pooling_descriptor const &) = delete;

};


}}}} // namespace znn::fwd::device::cudnn
