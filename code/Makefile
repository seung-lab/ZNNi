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

NVCC	=	$(AT)/usr/local/cuda/bin/nvcc

ifneq ($(strip $(ICC)),)
	CXX 	=	$(AT)icc
else
	CXX	=	$(AT)g++
endif

FPIC	=	-fPIC

CXXINCLUDES	=	-I$(HERE)/include -I$(HERE)/.. -I/usr/local/cuda/include
CXXWARN		=	-Wall -Wextra

DEPFLAGS	=	-std=c++11 -MM -MG -MP $(CXXINCLUDES) -MT "$(@:.d=.o)"
INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T)

COMMON_FLAGS	=	-c -g -std=c++11 $(INLINE_DEPFLAGS) $(CXXINCLUDES) $(FPIC) $(CXXWARN)

DBG_FLAGS		=	$(COMMON_FLAGS)
OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -mavx2 -mfma -ffast-math

OPT_FLAGS		=	$(COMMON_FLAGS) $(OPTIMIZATION_FLAGS)

COMMON_LDFLAGS		=	-g $(FPIC) -L/usr/local/cuda/lib -ltbb -ltbbmalloc -ltbbmalloc_proxy -lfftw3f -lcudart

DBG_LDFLAGS		=	$(COMMON_LDFLAGS)
OPT_LDFLAGS		=	$(COMMON_LDFLAGS) -O3 -DNDEBUG -mavx2 -mfma -ffast-math


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

# c++
$(BINDIR)/%.o: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -c $(CXXFLAGS) -o $@ $<
	$(MV) -f "$(@:.o=.T)" "$(@:.o=.d)"

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

remake: clean all
