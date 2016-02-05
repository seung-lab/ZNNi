#pragma once

#include "batch_divided_layer.hpp"
#include "input_output_divided_layer.hpp"
#include "simple_conv_layer.hpp"
#include "../cpu/cpu3d.hpp"

namespace znn { namespace fwd { namespace gpu3dram {


class conv_layer: public cpu3d::cpu_layer
{
private:
    simple_conv_layer*                simple_ = nullptr;
    batch_divided_layer*              batch_  = nullptr;
    batch_input_output_divided_layer* bio_    = nullptr;

    real * kernel_data_ ;
    real * bias_data_   ;

    long_t kernel_elements;
    long_t bias_elements;

    long_t total_in_elements;
    long_t total_out_elements;

public:

    real* kernel_data()
    {
        return kernel_data_;
    }

    real* bias_data()
    {
        return bias_data_;
    }

    ~conv_layer()
    {
        znn_free(kernel_data_);
        znn_free(bias_data_);
        if (simple_) delete simple_;
        if (batch_ ) delete batch_ ;
        if (bio_   ) delete bio_   ;
    }

    int in_memory() const override
    {
        return 0;
    }

    int out_memory() const override
    {
        return 0;
    }

    conv_layer( cudnnHandle_t& handle,
                int n, int fin, int fout,
                vec3i const & is,
                vec3i const & fs )
    {
        kernel_data_ = znn_malloc<real>(fin * fout * fs[0] * fs[1] * fs[2]);
        bias_data_   = znn_malloc<real>(fout);

        vec3i os = is - fs + vec3i::one;
        long_t input_elements  = is[0] * is[1] * is[2];
        long_t output_elements = os[0] * os[1] * os[2];

	total_in_elements = input_elements * n * fin;
	total_out_elements = output_elements * n * fout;

	kernel_elements = fs[0] * fs[1] * fs[2] * fin * fout;
	bias_elements = fout;

        total_out_elements = output_elements * fout * n;

        if ( input_elements * (fin+fout) * n < 500000000 )
        {
            simple_ = new simple_conv_layer(handle, n, fin, fout, is, fs);
        }
        else
        {
            if ( input_elements * (fin+fout) < 500000000 )
            {
                batch_ = new batch_divided_layer
                    (handle,
                     n,
                     n * input_elements * (fin+fout) / 500000000,
                     fin, fout, is, fs);
            }
            else
            {
                long_t bfin  = input_elements * fin / 250000000;
                long_t bfout = output_elements * fout / 250000000;

                bfin  = std::max(bfin , static_cast<long_t>(1));
                bfout = std::max(bfout, static_cast<long_t>(1));
                bio_ = new batch_input_output_divided_layer
                    (handle, n, fin, bfin, fout, bfout, is, fs);
            }
        }
    }

    real * forward( real * in ) override
    {
        real * out = znn_malloc<real>(total_out_elements);

        if ( simple_ )
        {
            float * workspace;
	    float * din;
	    float * dout;
	    float * dkernels;
	    float * dbiases;
	    
	    checkCudaErrors( cudaMalloc(&din,      total_in_elements * sizeof(float)     ));
	    checkCudaErrors( cudaMalloc(&dout,     total_out_elements * sizeof(float)    ));
	    checkCudaErrors( cudaMalloc(&dkernels, kernel_elements * sizeof(float) ));
	    checkCudaErrors( cudaMalloc(&dbiases,  bias_elements * sizeof(float)   ));

	    checkCudaErrors( cudaMemcpy(din, in, total_in_elements * sizeof(float), cudaMemcpyHostToDevice));
	    checkCudaErrors( cudaMemcpy(dkernels, kernel_data_, kernel_elements * sizeof(float), cudaMemcpyHostToDevice));
	    checkCudaErrors( cudaMemcpy(dbiases, bias_data_, bias_elements * sizeof(float), cudaMemcpyHostToDevice));
	    
            if ( simple_->workspace_memory() )
            {
                checkCudaErrors( cudaMalloc(&workspace, simple_->workspace_memory()));
            }

            simple_->forward(din, dout, dkernels, 0, workspace);
	    simple_->nonlinearity(dout, dbiases);

	    checkCudaErrors( cudaMemcpy(out, dout, total_out_elements * sizeof(float), cudaMemcpyDeviceToHost));
	    
            if ( simple_->workspace_memory() )
	      {
                checkCudaErrors( cudaFree(workspace) );
            }

	    checkCudaErrors( cudaFree(din));
	    checkCudaErrors( cudaFree(dout));
	    checkCudaErrors( cudaFree(dkernels));
	    checkCudaErrors( cudaFree(dbiases));

        }
        else
        {
            if ( batch_ ) batch_->forward(in, out, kernel_data_, bias_data_);
            if ( bio_ ) bio_->forward(in, out, kernel_data_, bias_data_);
        }

        znn_free(in);
        return out;
    }

};


}}} // namespace znn::fwd::gpu3dram
