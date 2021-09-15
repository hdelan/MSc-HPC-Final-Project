SHELL 		:= /bin/bash
CXX 		:= g++
NVCXX 		:= nvcc
NVCCFLAGS 	:= --use_fast_math -Wno-deprecated-gpu-targets -rdc=true #-arch=sm_35
CXXFLAGS 	:= -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS 	 	:= -lopenblas -llapack -llapacke -lm -lcublas -lpthread -lcudadevrt #"-I/usr/local/opt/lapack/include"
OPT			:= -O3

SRCDIR 		:= ./lib
BUILDDIR 	:= ./build
TESTDIR 	:= ./tests

FILES 		:= adjMatrix.cc \
			make_graph.cc \
                        lanczos.cu

                     

CUFILES		:= cu_lanczos.cu \
			cu_SPMV.cu \
		      	cu_linalg.cu \
                        check_ans.cu \
       			multiplyOut.cu \
       			eigen.cu \
                        helpers.cu

TESTOBJS 	:=  $(BUILDDIR)/cu_linalg.o \
				$(BUILDDIR)/cu_SPMV.o \
				$(BUILDDIR)/cu_lanczos.o \
				$(BUILDDIR)/lanczos.o \
				$(BUILDDIR)/SPMV.o \
				$(BUILDDIR)/helpers.o \
				$(BUILDDIR)/adjMatrix.o \
				$(BUILDDIR)/check_ans.o \
				$(BUILDDIR)/make_graph.o \
				$(BUILDDIR)/eigen.o \
				$(BUILDDIR)/multiplyOut.o \
				$(BUILDDIR)/cu_multiplyOut.o

OBJS		:= $(FILES:%.cc=$(BUILDDIR)/%.o) $(CUFILES:%.cu=$(BUILDDIR)/%.o) 

PROFILE 	:= 

########################################################################

all:  linalg_test SPMV_test SPMV_blocks lanczos_test

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc $(SRCDIR)/%.h
	$(NVCXX) $(NVCCFLAGS) $(OPT) $(PROFILE) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/%.h
	$(NVCXX) $(NVCCFLAGS) $(OPT) $(PROFILE) -c $< -o $@

.PHONY:
	clean run 

clean:
	rm $(BUILDDIR)/* *test

linalg_test: $(TESTDIR)/linalg_test.cu $(TESTOBJS) lib/blocks.h Makefile 
	$(NVCXX) $(TESTDIR)/linalg_test.cu $(TESTOBJS) $(NVCCFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@
	./linalg_test
        

SPMV_test: $(TESTDIR)/SPMV_test.cu $(TESTOBJS) lib/blocks.h Makefile 
	$(NVCXX) $(TESTDIR)/SPMV_test.cu $(TESTOBJS) $(NVCCFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@
	./SPMV_test

SPMV_blocks: $(TESTDIR)/SPMV_blocks.cu $(TESTOBJS) lib/blocks.h Makefile 
	$(NVCXX) $(TESTDIR)/SPMV_blocks.cu $(TESTOBJS) $(NVCCFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@
	./SPMV_blocks
        

lanczos_test: $(TESTDIR)/lanczos_test.cu $(TESTOBJS) lib/blocks.h Makefile 
	$(NVCXX) $(TESTDIR)/lanczos_test.cu $(TESTOBJS) $(NVCCFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@
	./lanczos_test

