/usr/local/cuda/bin/nvcc -c -std=c++11 -O3 -DNDEBUG -I../../ -I../include ../include/gpu/convolutional/cufft/utils.cu -o utils.o -I/usr/people/zlateski/cuda/include
