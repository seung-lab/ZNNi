#pragma once

#include "base.hpp"
#include "native_conv3d.hpp"

namespace znn { namespace fwd { namespace gpu3dram {

class full_conv3d: public base_conv3d
{
private:
    native_conv3d native_;

public:
    ~full_conv3d() {}

    void forward( float * in,
                  float * out,
                  float * kernels,
                  float * biases ) const override
    {
        float * workspace = NULL ;
        float * in_d      ;
        float * out_d     ;
        float * kernels_d ;
        float * biases_d  ;

        if ( native_.workspace_memory() )
        {
            checkCudaErrors( cudaMalloc(&workspace,
                                        native_.workspace_memory()));
        }

        checkCudaErrors( cudaMalloc(&in_d     , native_.in_memory()     ));
        checkCudaErrors( cudaMalloc(&out_d    , native_.out_memory()    ));
        checkCudaErrors( cudaMalloc(&kernels_d, native_.kernel_memory() ));
        checkCudaErrors( cudaMalloc(&biases_d , native_.bias_memory()   ));

        checkCudaErrors( cudaMemcpy(in_d, in, native_.in_memory(),
                                    cudaMemcpyHostToDevice));
        checkCudaErrors( cudaMemcpy(kernels_d, kernels, native_.kernel_memory(),
                                    cudaMemcpyHostToDevice));
        checkCudaErrors( cudaMemcpy(biases_d, biases, native_.bias_memory(),
                                    cudaMemcpyHostToDevice));

        native_.forward(in_d, out_d, kernels_d, 0, workspace);
        native_.nonlinearity(out_d, biases_d);

        checkCudaErrors( cudaMemcpy(out, out_d, native_.out_memory(),
                                    cudaMemcpyDeviceToHost));


        if ( native_.workspace_memory() )
        {
            checkCudaErrors( cudaFree(workspace) );
        }

        checkCudaErrors( cudaFree(in_d));
        checkCudaErrors( cudaFree(out_d));
        checkCudaErrors( cudaFree(kernels_d));
        checkCudaErrors( cudaFree(biases_d));

    }


public:
    full_conv3d( cudnnHandle_t& handle,
                 long_t n,
                 long_t fin,
                 long_t fout,
                 vec3i const & is,
                 vec3i const & fs )
        : native_(handle, n,fin,fout,is,fs)
    {}


};

}}} // namespace znn::fwd::gpu3dram
