ODIR		=	.
SFL 		=	znni.cpp
CPP 		=	g++
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I./ -I../include -I../include/znn/util -I../../ -I/usr/local/cuda-7.5/include -I/usr/local/cuda-7.0/include -I/opt/intel/tbb/include -I/usr/local/cudnn-6.5-linux-x64-v2/
LIB_FLAGS	=	-L./ -L/usr/local/cuda-7.0/lib64 -L/opt/intel/tbb/lib/intel64/gcc4.4 -L/usr/local/cudnn-6.5-linux-x64-v2/
OPT_FLAGS	=	-O0 -std=c++1y
MKL_FLAGS       =       -static-intel -mkl=sequential -DZNN_USE_MKL_FFT -DZNN_USE_MKL_NATIVE_FFT
OTH_FLAGS	=	-Wall -Wno-unused-result -Wno-unused-local-typedefs
LIBS		=	-lpthread -pthread -ltbb -ltbbmalloc -ltbbmalloc_proxy -lfftw3f -lcudart -lcufft -lcudnn -lcublas -lhdf5_cpp -lhdf5

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBS += -lrt
endif

znni: $(SFL)
	$(CPP) -o $(ODIR)/znni $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

.PHONY: mkl clean

mkl: $(SFL)
	/opt/intel/bin/icc -o $(ODIR)/znni $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(MKL_FLAGS) $(OTH_FLAGS) $(LIBS)

clean:
	rm -f $(ODIR)/libznni.so
