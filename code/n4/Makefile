ODIR		=	.
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I./ -I../include -I../include/znn/util -I../../ -I/usr/local/cuda/include -I/opt/intel/tbb/include
LIB_FLAGS	=	-L./ -L/usr/local/cuda/lib64 -L/opt/intel/tbb/lib/intel64/gcc4.4
NVCC_FLAGS	= 	#-DNDEBUG
OPT_FLAGS	=	-O3  -std=c++11
CPU_FLAGS	=	-ffast-math -Wall -Wno-unused-result -Wno-unused-local-typedefs
MKL_FLAGS	=	-static-intel -mkl=sequential -DZNN_USE_MKL_FFT -DZNN_USE_MKL_NATIVE_FFT -DZNN_USE_MKL_CONVOLUTION
LIBS		=	-ltbb -ltbbmalloc -ltbbmalloc_proxy -lfftw3f  -lhdf5_cpp -lhdf5
CPU_LIBS	=	-lpthread -pthread
GPU_LIBS	=	-lcudart -lcufft -lcudnn -lcublas

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBS += -lrt
endif

znni: znni_gpu.cpp
	/usr/local/cuda/bin/nvcc -o $(ODIR)/znni znni_gpu.cpp $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(NVCC_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)  $(GPU_LIBS)

.PHONY: cpu mkl gpu clean

cpu: znni_cpu.cpp
	g++ -o $(ODIR)/znni znni_cpu.cpp $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS)  $(CPU_FLAGS)  $(LIBS)  $(CPU_LIBS)


mkl: znni_cpu.cpp
	/opt/intel/bin/icc -o $(ODIR)/znni znni_cpu.cpp $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS)  $(CPU_FLAGS)  $(MKL_FLAGS) $(LIBS)  $(CPU_LIBS)



clean:
	rm -f $(ODIR)/libznni.so
