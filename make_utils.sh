nvcc -c -std=c++11 -O3 -DNDEBUG -I. -I./src/include ./src/include/gpu_fft/utils.cu -o utils.o
