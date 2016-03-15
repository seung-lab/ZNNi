# -*- Makefile -*-

HERE    	=       .
BINDIR		=	./bin

AT	=       @
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
	CXX 			=	$(AT)icc
	OPTIMIZATION_FLAGS	=	-DNDEBUG -O3
	CS_FLAGS		=	-static-intel -mkl=sequential
else
	CXX			=	$(AT)g++
	OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -mavx2 -mfma -ffast-math
	CS_FLAGS		=
endif

FPIC	=	-fPIC

CXXINCLUDES	=	-I$(HERE)/include -I$(HERE)/.. -I/usr/local/cuda/include
CXXWARN		=	-Wall -Wextra

NVCCFLAGS	=	-std=c++11 -O3 -DNDEBUG $(CXXINCLUDES)

DEPFLAGS	=	-std=c++11 -MM -MG -MP $(CXXINCLUDES) -MT "$(@:.d=.o)"
INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T)

COMMON_FLAGS	=	-c -g -std=c++11 $(INLINE_DEPFLAGS) $(CXXINCLUDES) $(FPIC) $(CXXWARN)

DBG_FLAGS		=	$(COMMON_FLAGS)
#OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -mavx2 -mfma -ffast-math

OPT_FLAGS		=	$(COMMON_FLAGS) $(OPTIMIZATION_FLAGS) $(CS_FLAGS)

COMMON_LDFLAGS		=	-g $(FPIC) -L/usr/local/cuda/lib64 -ltbb -ltbbmalloc -ltbbmalloc_proxy -lfftw3f -lcudart -lcufft -lcudnn -lcublas

DBG_LDFLAGS		=	$(COMMON_LDFLAGS)
OPT_LDFLAGS		=	$(COMMON_LDFLAGS) $(OPTIMIZATION_FLAGS) $(CS_FLAGS)

ifneq ($(strip $(OPT)),)
  CXXFLAGS	=	$(OPT_FLAGS)
  CXXLDFLAGS	=	$(OPT_LDFLAGS)
else
  CXXFLAGS	=	$(DBG_FLAGS)
  CXXLDFLAGS	=	$(DBG_LDFLAGS)
endif

OBJS = test.o benchmark_host.o

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

$(BINDIR)/test: $(BINDIR)/test.o
	$(ECHO) "[CXX] linking $<"
	$(CXX) $< $(CXXLDFLAGS) -o $@

$(BINDIR)/benchmark_host: $(BINDIR)/benchmark_host.o
	$(ECHO) "[CXX] linking $<"
	$(CXX) $< $(CXXLDFLAGS) -o $@

$(BINDIR)/benchmark_device: $(BINDIR)/benchmark_device.o $(BINDIR)/kernels.o
	$(ECHO) "[CXX] linking $<"
	$(CXX) $< $(CXXLDFLAGS) -o $@

remake: clean all
