#include <cufft.h>
#include <fftw3.h>
#include <cublas_v2.h>
#include "gpu/gpu2d.hpp"
#include "gpu/gpu3d.hpp"
#include "descriptor2.hpp"
#include <boost/multi_array.hpp>

using namespace znn::fwd;


void gemv(cublasHandle_t cublasHandle, int m, int n, float alpha,
          const float *A, const float *x,
          float beta, float *y)
{
#ifdef DISABLE_GEMV
    checkCublasErrors( cublasSgemm (cublasHandle,
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      n,
                      1,
                      m,
                      &alpha,
                      A,
                      m,
                      x,
                      m,
                      &beta,
                      y,
                      m) );
#else
    checkCublasErrors( cublasSgemv(cublasHandle, CUBLAS_OP_N,
                                  m, n,
                                  &alpha,
                                  A, m,
                                  x, 1,
                                  &beta,
                                  y, 1) );
#endif
};


template<typename T>
inline void examine(T* p, size_t len, size_t len2)
{
    T x[len*len2];

    cudaMemcpy( x, p, len*len2 * sizeof(T), cudaMemcpyDeviceToHost);

    for ( size_t i  = 0; i < len*len2; ++i )
    {
        if ( i % len == 0 )
            std::cout << "\n";
        std::cout << x[i] << ' ';
    }

    std::cout << "\n\n";
}

template<typename T>
inline void examineh(T* p, size_t len, size_t len2)
{
    for ( size_t i  = 0; i < len*len2; ++i )
    {
        if ( i % len == 0 )
            std::cout << "\n";
        std::cout << p[i] << ' ';
    }

    std::cout << "\n\n";
}


int main()
{

    int version = (int)cudnnGetVersion();
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n",
           version, CUDNN_VERSION, CUDNN_VERSION_STR);
    printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
    showCudaDevices();

    int device = 0;
    checkCudaErrors( cudaSetDevice(device) );
    std::cout << "Using device " << device << std::endl;

    struct cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties( &prop, device ));
    double globalMem = prop.totalGlobalMem/double(1024*1024);

    std::cout << "Memory: " << globalMem << std::endl;

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    boost::multi_array<float, 2> mat(boost::extents[5][6]);
    for ( int i = 0; i < 5*6; ++i ) mat.data()[i] = i;

    examineh(mat.data(), 5, 6);

    boost::multi_array<float, 2> vec(boost::extents[1][6]);
    for ( int i = 0; i < 6; ++i ) vec.data()[i] = 1;

    examineh(vec.data(), 1, 6);

    float* matd; cudaMalloc(&matd, 30*sizeof(float));
    float* vecd; cudaMalloc(&vecd, 6*sizeof(float));
    float* res; cudaMalloc(&res, 5*sizeof(float));

    cudaMemcpy(matd, mat.data(), 30*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vecd, vec.data(), 6*sizeof(float), cudaMemcpyHostToDevice);

    examine(matd, 5, 6);
    examine(vecd, 1, 6);


    gemv(cublasHandle, 5, 6, 1, matd, vecd, 0, res);

    examine(res, 1, 5);

    cudaDeviceReset();

    return 0;


#define NX 34
#define NY 71
#define NZ 123
#define CZ (NZ/2+1)
#define NN 20

    real*          in_h  = znn_malloc<real>(NX*NY*NZ*NN);
    fftwf_complex* out_h = znn_malloc<fftwf_complex>(NX*NY*CZ*NN);
    fftwf_complex* out2_h = znn_malloc<fftwf_complex>(NX*NY*CZ*NN);

    int dims[3] = {NX, NY, NZ};
    fftwf_plan fft_plan =
        fftwf_plan_many_dft_r2c( 3, dims, NN,
                                 in_h, NULL, 1, NX*NY*NZ,
                                 out_h, NULL, 1, NX*NY*CZ,
                                 FFTW_ESTIMATE );


    cufftComplex *out_d;
    cufftReal    *in_d;


    uniform_init ia(0.1);
    ia.initialize(in_h, NX*NY*NZ);


    cufftHandle plan;
    cufftPlanMany(&plan, 3, dims, NULL, 0,
                  NX*NY*NZ, NULL, 0,
                  CZ*NY*NX,
                  CUFFT_R2C, 20);


    zi::wall_timer wt;

    wt.reset();
    checkCudaErrors(cudaMalloc(&in_d, sizeof(cufftReal)*NX*NY*NZ*NN));
    checkCudaErrors(cudaMalloc(&out_d, sizeof(cufftComplex)*NX*NY*CZ*NN));
    checkCudaErrors( cudaMemset(in_d, 0, sizeof(real)*NX*NY*NZ*20));
    checkCudaErrors( cudaMemcpy(in_d, in_h,
                                sizeof(real)*NX*NY*NZ*20,
                                cudaMemcpyHostToDevice) );
    cufftExecR2C(plan, in_d, out_d);

    checkCudaErrors( cudaMemcpy(out2_h, out_d,
                                sizeof(cufftComplex)*NX*NY*CZ*NN,
                                cudaMemcpyDeviceToHost) );

    std::cout << "GPU TIME: " << wt.elapsed<double>() << std::endl;
    cudaFree(in_d); cudaFree(out_d);

    wt.reset();

    fftwf_execute_dft_r2c(fft_plan, in_h, out_h);

    std::cout << "CPU TIME: " << wt.elapsed<double>() << std::endl;


    real* r1 = reinterpret_cast<real*>(out_h);
    real* r2 = reinterpret_cast<real*>(out2_h);

    real ma = 0;
    real m1 = 0;

    for ( size_t i = 0; i < NX*NY*CZ*NN*2; ++i )
    {
        ma = std::max(ma, std::abs(r1[i] - r2[i]));
        m1 = std::max(m1, std::abs(r1[i]));
    }

    std::cout << ma << "\n";
    std::cout << m1 << "\n";

    cufftDestroy(plan);

    cudaDeviceReset();


}
