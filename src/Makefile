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

INCLUDES	=	-I$(HERE)/include -I$(HERE)/.. -I/usr/local/cuda/include
WARN		=	-Wall -Wextra
DEPFLAGS	=	-MM -MG -MP $(INCLUDES) -MT "$(@:.d=.o)"
INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T)

COMMON_FLAGS	=	-c -g -std=c++11 $(INLINE_DEPFLAGS) $(INCLUDES) $(FPIC) $(WARN)


DBG_FLAGS		=	$(COMMON_FLAGS)
OPTIMIZATION_FLAGS	=	-DNDEBUG -O3

OPT_FLAGS		=	$(COMMON_FLAGS) $(OPTIMIZATION_FLAGS)

COMMON_LDFLAGS		=	-g $(FPIC) -Wl,--eh-frame-hdr -lm

DBG_LDFLAGS		=	$(COMMON_LDFLAGS)
OPT_LDFLAGS		=	$(COMMON_LDFLAGS) -O3


ifneq ($(strip $(OPT)),)
  FLAGS		=	$(OPT_FLAGS)
  LDFLAGS	=	$(OPT_LDFLAGS)
else
  FLAGS		=	$(DBG_FLAGS)
  LDFLAGS	=	$(DBG_LDFLAGS)
endif


# dependency files for c++
$(BINDIR)/%.d: ./%.cpp
	$(MKDIR) -p $(dir $@)
	$(CXX) $(DEPFLAGS) -MF $@ $<

# c++
$(BINDIR)/%.o: ./%.cpp $(BINDIR)/%.d
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -c $(FLAGS) -o $@ $<
	$(MV) -f "$(@:.o=.T)" "$(@:.o=.d)"

.PHONY: all
all: $(BINDIR)/cpp/test_cpu.o

.PHONY: clean
clean:
	$(ECHO) Cleaning...
	$(RM) -rf $(BINDIR)

.PHONY: remake
remake: clean all
