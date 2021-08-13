SHELL 		:= /bin/bash
CXX 		:= g++
NVCXX 		:= nvcc
NVCCFLAGS 	:= --use_fast_math -Wno-deprecated-gpu-targets #-arch=sm_35 
CXXFLAGS 	:= -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS 	 	:= -lblas -llapack -llapacke -lm #"-I/usr/local/opt/lapack/include"
OPT			:= -O3

MAIN 		:= adj

SRCDIR 		:= ./lib
BUILDDIR 	:= ./build
TESTDIR 	:= ./tests

FILES 		:= adjMatrix.cc \
       			helpers.cc \
			make_graph.cc

CUFILES		:= cu_lanczos.cu \
			cu_SPMV.cu \
		      	cu_linalg.cu \
       			multiplyOut.cu \
       			eigen.cu 

TESTOBJS 	:=  $(BUILDDIR)/cu_linalg.o \
				$(BUILDDIR)/cu_SPMV.o \
				$(BUILDDIR)/cu_lanczos.o \
				$(BUILDDIR)/SPMV.o \
				$(BUILDDIR)/helpers.o \
				$(BUILDDIR)/adjMatrix.o \
				$(BUILDDIR)/make_graph.o \
				$(BUILDDIR)/eigen.o \
				$(BUILDDIR)/multiplyOut.o \

OBJS		:= $(FILES:%.cc=$(BUILDDIR)/%.o) $(CUFILES:%.cu=$(BUILDDIR)/%.o) 

PROFILE 	:= 

########################################################################

$(MAIN): main.cu $(OBJS) 
	$(NVCXX) $^ $(NVCCFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc $(SRCDIR)/%.h
	$(NVCXX) $(NVCCFLAGS) $(OPT) $(PROFILE) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/%.h
	$(NVCXX) $(NVCCFLAGS) $(OPT) $(PROFILE) -c $< -o $@

.PHONY:
	clean run 

clean:
	rm $(BUILDDIR)/* $(MAIN) *.out 

run: $(MAIN)
	./adj 

linalg_test: $(TESTDIR)/linalg_test.cu $(TESTOBJS)
	$(NVCXX) $^ $(NVCCFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@

lanczos_test: $(TESTDIR)/lanczos_test.cu $(TESTOBJS)
	$(NVCXX) $^ $(NVCCFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@
