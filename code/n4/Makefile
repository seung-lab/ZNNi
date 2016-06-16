ODIR				=		.
SFL 				=		znni.cpp
CPP_FLAGS		= 	-g
INC_FLAGS		=		-I./ -I../include -I../include/znn/util -I../../ -I/usr/local/cuda/include -I/opt/intel/tbb/include -I/usr/local/cudnn-6.5-linux-x64-v2/
LIB_FLAGS		=		-L./ -L/usr/local/cuda/lib64 -L/opt/intel/tbb/lib/intel64/gcc4.4 -L/usr/local/cudnn-6.5-linux-x64-v2/
NVCC_FLAGS	= 	-DNDEBUG
OPT_FLAGS		=		-O3  -std=c++11 -ffast-math
MKL_FLAGS		=		-static-intel -mkl=sequential -DZNN_USE_MKL_FFT -DZNN_USE_MKL_NATIVE_FFT -DZNN_USE_MKL_CONVOLUTION
OTH_FLAGS		=		-Wall -Wno-unused-result -Wno-unused-local-typedefs
LIBS				=		-lpthread -pthread -ltbb -ltbbmalloc -ltbbmalloc_proxy -lfftw3f -lcudart -lcufft -lcudnn -lcublas -lhdf5_cpp -lhdf5

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBS += -lrt
endif

znni: znni_gpu.cpp
	/usr/local/cuda/bin/nvcc -o $(ODIR)/znni $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

.PHONY: cpu mkl clean

cpu: znni_cpu.cpp
	g++ -o $(ODIR)/znni znni_cpu.cpp $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)


mkl: znni_cpu.cpp
	/opt/intel/bin/icc -o $(ODIR)/znni znni_cpu.cpp $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(MKL_FLAGS) $(OTH_FLAGS) $(LIBS)

clean:
	rm -f $(ODIR)/libznni.so
