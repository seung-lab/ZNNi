# -*- Makefile -*-

HERE    	=       .
BINDIR		=	./bin

AT	=	@
DOLLAR  = 	$$

CD	=	$(AT)cd
CP	=	$(AT)cp
ECHO	=	@echo
CAT	=	$(AT)cat
IF	=	$(AT)if
LN	=	$(AT)ln
MKDIR	=	$(AT)mkdir
MV	=	$(AT)mv
SED	=	$(AT)sed
RM	=	$(AT)rm
TOUCH	=	$(AT)touch
CHMOD	=	$(AT)chmod
DATE    =       $(AT)date

# compilers

NVCC		=	$(AT)/usr/local/cuda/bin/nvcc

ifneq ($(strip $(ICC)),)
	CXX 			=	$(AT)icpc
	OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -ipo
	CS_FLAGS		=	-DZNN_USE_MKL_FFT
	CS_LD_FLAGS		=	-static-intel -mkl=sequential
else
	CXX			=	$(AT)g++
	OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -ffast-math
	CS_FLAGS		=
	CS_LD_FLAGS		=
endif

FPIC	=	-fPIC

CXXINCLUDES	=	-I$(HERE)/include -I$(HERE)/.. -I/usr/local/cuda/include
CXXWARN		=	-Wall -Wextra -Wno-format

NVCCFLAGS	=	-std=c++11 -O3 -DNDEBUG $(CXXINCLUDES)

DEPFLAGS	=	-std=c++11 -MM -MG -MP $(CXXINCLUDES) -MT "$(@:.d=.o)"
INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T)

COMMON_FLAGS	=	-g -std=c++11 $(INLINE_DEPFLAGS) $(CXXINCLUDES) $(CXXWARN)

DBG_FLAGS		=	$(COMMON_FLAGS)
LD_OPTIMIZATION_FLAGS	=	-DNDEBUG -O3

OPT_FLAGS		=	$(COMMON_FLAGS) $(OPTIMIZATION_FLAGS) $(CS_FLAGS)

COMMON_LDFLAGS		=	-g -L/usr/local/cuda/lib64 -ltbb -ltbbmalloc -ltbbmalloc_proxy -lfftw3f -lcudart -lcufft -lcudnn -lcublas

DBG_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS)
OPT_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) $(LD_OPTIMIZATION_FLAGS)

ifneq ($(strip $(OPT)),)
  CXXFLAGS	=	$(OPT_FLAGS)
  CXXLDFLAGS	=	$(OPT_LDFLAGS)
else
  CXXFLAGS	=	$(DBG_FLAGS)
  CXXLDFLAGS	=	$(DBG_LDFLAGS)
endif

OBJS = test.o test_host.o test_device.o benchmark_device.o \
	benchmark_host.o benchmark_baseline.o kernels.o \
	benchmark_device_optimal.o benchmark_best_device.o

DEPS = $(patsubst %.o,$(BINDIR)/%.d,$(OBJS))

-include $(DEPS)

# dependency files for c++
$(BINDIR)/%.d: ./%.cpp
	$(ECHO) "[CXX] dependencies $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) $(DEPFLAGS) -MF $@ $<

# dependency files for c++
$(BINDIR)/%.d: ./%.cu
	$(ECHO) "[CXX] dependencies $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -x c++ $(DEPFLAGS) -MF $@ $<

# c++
$(BINDIR)/%.o: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -c $(CXXFLAGS) -o $@ $<
	$(MV) -f "$(@:.o=.T)" "$(@:.o=.d)"

# cu
$(BINDIR)/%.o: ./%.cu
	$(ECHO) "[NVCC] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -x c++ $(DEPFLAGS) -MF $@ $<
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<


.PHONY: all clean remake

all:
	$(ECHO) $(DEPS)

clean:
	$(ECHO) Cleaning...
	$(RM) -rf $(BINDIR)

$(BINDIR)/test: $(BINDIR)/kernels.o $(BINDIR)/test.o
	$(ECHO) "[CXX] linking $^"
	$(CXX) $^ $(CXXLDFLAGS) -o $@

$(BINDIR)/test_host: $(BINDIR)/test_host.o
	$(ECHO) "[CXX] linking $<"
	$(CXX) $< $(CXXLDFLAGS) -o $@

$(BINDIR)/test_device: $(BINDIR)/kernels.o $(BINDIR)/test_device.o
	$(ECHO) "[CXX] linking $^"
	$(CXX) $^ $(CXXLDFLAGS) -o $@

$(BINDIR)/benchmark_host: $(BINDIR)/benchmark_host.o
	$(ECHO) "[CXX] linking $^"
	$(CXX) $< $(CXXLDFLAGS) -o $@

$(BINDIR)/benchmark_device:  $(BINDIR)/kernels.o $(BINDIR)/benchmark_device.o
	$(ECHO) "[CXX] linking $^"
	$(CXX) $^ $(CXXLDFLAGS) -o $@

$(BINDIR)/benchmark_baseline:  $(BINDIR)/kernels.o $(BINDIR)/benchmark_baseline.o
	$(ECHO) "[CXX] linking $^"
	$(CXX) $^ $(CXXLDFLAGS) -o $@

$(BINDIR)/benchmark_device_optimal:  $(BINDIR)/kernels.o $(BINDIR)/benchmark_device_optimal.o
	$(ECHO) "[CXX] linking $^"
	$(CXX) $^ $(CXXLDFLAGS) -o $@

$(BINDIR)/benchmark_best_device:  $(BINDIR)/kernels.o $(BINDIR)/benchmark_best_device.o
	$(ECHO) "[CXX] linking $^"
	$(CXX) $^ $(CXXLDFLAGS) -o $@

remake: clean all
