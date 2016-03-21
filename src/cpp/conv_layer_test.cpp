#include "cpu/pooling/naive.hpp"
#include "cpu/pooling/default.hpp"
#include "cpu/convolutional/direct.hpp"
#include "cpu/convolutional/naive.hpp"
#include "cpu/convolutional/padded_pruned_fft.hpp"
#include "gpu/convolutional/cudnn.hpp"
#include "gpu/convolutional/cufft.hpp"
#include "gpu/convolutional/padded_cufft.hpp"
#include "gpu/convolutional/padded_pruned_cufft.hpp"
#include "init.hpp"
#include <zi/time.hpp>

using namespace znn::fwd;

void test_conv_layer()
{
  uniform_init ui(0.1);

  detail::random_number_generator_impl& rng =
      zi::singleton<detail::random_number_generator_impl>::instance();

  std::uniform_int_distribution<long_t> intdist(1,7);

  vec3i ws(intdist(rng.rng),intdist(rng.rng),intdist(rng.rng));

  std::uniform_int_distribution<long_t> intdist2(19,29);

  vec3i os(intdist2(rng.rng),intdist2(rng.rng),intdist2(rng.rng));

  std::uniform_int_distribution<long_t> intdist3(3,7);

  long_t n = intdist3(rng.rng);
  long_t l = intdist3(rng.rng);
  long_t l2 = intdist3(rng.rng);

  vec3i is = os + ws - vec3i::one;

  host_array<real> kernels = get_array<real>(l*l2*ws[0]*ws[1]*ws[2]);
  host_array<real> biases  = get_array<real>(l2);

  ui.initialize(kernels.get(), l*l2*ws[0]*ws[1]*ws[2]);
  ui.initialize(biases.get(), l2);

  //cpu::naive_convolutional_layer  np(n,l,l2,is,ws, kernels.get(), biases.get());
  cpu::direct_convolutional_layer np(n,l,l2,is,ws, kernels.get(), biases.get());
  cpu::padded_pruned_fft_convolutional_layer pl(n,l,l2,is,ws, kernels.get(), biases.get());
  gpu::cudnn_convolutional_layer cudn(n,l,l2,is,ws, kernels.get(), biases.get());
  gpu::cufft_convolutional_layer cudf(n,l,l2,is,ws, kernels.get(), biases.get());
  gpu::padded_cufft_convolutional_layer cudfp(n,l,l2,is,ws, kernels.get(), biases.get());
  gpu::padded_pruned_cufft_convolutional_layer cudfpp(n,l,l2,is,ws, kernels.get(), biases.get());

  host_array<real> ina = get_array<real>(np.total_input_len);
  ui.initialize(ina.get(), np.total_input_len);

  host_array<real> inb = get_array<real>(np.total_input_len);
  std::copy_n(ina.get(), np.total_input_len, inb.get());

  host_array<real> inx = get_array<real>(np.total_input_len);
  std::copy_n(ina.get(), np.total_input_len, inx.get());

  device_array<real> inc = get_device_array<real>(np.total_input_len);
  device_copy_n(ina.get(), np.total_input_len, inc);

  device_array<real> ine = get_device_array<real>(np.total_input_len);
  device_copy_n(ina.get(), np.total_input_len, ine);

  // device_array<real> inf = get_device_array<real>(np.total_input_len);
  // device_copy_n(ina.get(), np.total_input_len, inf);

  std::cout << "   NETL " << ws << ' ' << n << ' '
	    << l << ' ' << l2 << ' ' << os << "\n";

  zi::wall_timer wt;
  auto r1 = np.forward(std::move(ina));
  std::cout << "    Direct took        : " << wt.elapsed<double>() << std::endl;

  // wt.reset();
  // auto r1 = pl.forward(std::move(ina));
  // std::cout << "    FFT took           : " << wt.elapsed<double>() << std::endl;

  wt.reset();
  auto r2 = pl.forward(std::move(inb));
  std::cout << "    FFT took           : " << wt.elapsed<double>() << std::endl;

  wt.reset();
  auto r3 = cudn.forward(std::move(inc));
  std::cout << "    GPU took           : " << wt.elapsed<double>() << std::endl;

  r3.reset();

  device_array<real> ind = get_device_array<real>(np.total_input_len);
  device_copy_n(inx.get(), np.total_input_len, ind);

  // wt.reset();
  // auto r4 = cudf.forward(std::move(ind));
  // std::cout << "    GPU FFT took       : " << wt.elapsed<double>() << std::endl;

  wt.reset();
  auto r5 = cudfp.forward(std::move(ine));
  std::cout << "    PADDED GPU FFT took: " << wt.elapsed<double>() << std::endl;

  wt.reset();
  auto r6 = cudfpp.forward(std::move(ind));
  std::cout << "    PRUNED GPU FFT took: " << wt.elapsed<double>() << std::endl;

  checkCudaErrors( cudaMemcpy(r2.get(), r6.get(),
   			      np.total_output_len * sizeof(float),
			      cudaMemcpyDeviceToHost) );

  // checkCudaErrors( cudaMemcpy(r1.get(), r3.get(),
  // 			      np.total_output_len * sizeof(float),
  // 			      cudaMemcpyDeviceToHost) );


  real mdiff = 0;

  bool ok = true;

  for ( long_t i = 0; i < np.total_output_len; ++i )
    {
      mdiff = std::max(mdiff,std::abs(r1.get()[i]- r2.get()[i]));

      if ( std::abs(r1.get()[i]- r2.get()[i]) > 0.0001 )
        {
	  //std::cout << "     " << r1.get()[i] << ' ' << r2.get()[i] << "\n";
        }

      ok &=  nearly_equal(r1.get()[i], r2.get()[i], 1000000);
    }

  std::cout << ( mdiff ) << "\n";

  //checkCUDNN( cudnnDestroy(gpu_handle) );

}

int main()
{
  int version = (int)cudnnGetVersion();
  printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
  printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
  showCudaDevices();

  int device = 0;
  checkCudaErrors( cudaSetDevice(device) );
  std::cout << "Using device " << device << std::endl;

  struct cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties( &prop, device ));
  double globalMem = prop.totalGlobalMem/double(1024*1024);

  std::cout << "Memory: " << globalMem << std::endl;

  while (1)
    test_conv_layer();

  cudaDeviceReset();

}
