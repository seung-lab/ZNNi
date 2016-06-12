ODIR		=	.
SFL 		=	znni.cpp
CPP 		=	g++
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I./ -I../include -I../../ -I/usr/local/cuda-7.5/include -I/usr/local/cuda-7.0/include -I/opt/intel/tbb/include
LIB_FLAGS	=	-L./ -L/usr/local/cuda-7.0/lib64 -L/opt/intel/tbb/lib/intel64/gcc4.4
OPT_FLAGS	=	-O3 -std=c++1y
OTH_FLAGS	=	-Wall -Wno-unused-result -Wno-unused-local-typedefs
LIBS		=	-lpthread -pthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBS += -lrt
endif

znni: $(SFL)
	$(CPP) -o $(ODIR)/znni $(SFL) $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS)

.PHONY: clean
clean:
	rm -f $(ODIR)/libznni.so
