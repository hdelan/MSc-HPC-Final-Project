SHELL 		:= /bin/bash
CXX 		:= g++
NVCXX 		:= nvcc
NVCCFLAGS 	:= --use_fast_math -arch=sm_35 -Wno-deprecated-gpu-targets
CXXFLAGS 	:= -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS 	 	:= -lblas -llapack -llapacke -lm #"-I/usr/local/opt/lapack/include"
OPT			:= -O3

MAIN 		:= adj

SRCDIR 		:= ./lib
BUILDDIR 	:= ./build
TESTDIR 	:= ./tests

FILES 		:= adjMatrix.cc \
       			sparse_mult.cc \
       			helpers.cc \
				make_graph.cc \

CUFILES		:= cu_lanczos.cu \
       			multiplyOut.cu \
       			eigen.cu \
				cu_SPMV.cu \
				linalg.cu
TESTOBJS 	:=  $(BUILDDIR)/cu_linalg.o \
				$(BUILDDIR)/cu_SPMV.o \
				$(BUILDDIR)/SPMV.o \
				$(BUILDDIR)/helpers.o \
				$(BUILDDIR)/adjMatrix.o \
				$(BUILDDIR)/make_graph.o \

OBJS		:= $(FILES:%.cc=$(BUILDDIR)/%.o) $(CUFILES:%.cu=$(BUILDDIR)/%.o) 

PROFILE 	:= 

########################################################################

$(MAIN): main.cu $(OBJS) 
	$(NVCXX) $^ $(LIBS) $(NVCCFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc $(SRCDIR)/%.h
	$(NVCXX) $(NVCCFLAGS) $(OPT) $(PROFILE) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu $(SRCDIR)/%.h
	$(NVCXX) $(NVCCFLAGS) $(OPT) $(PROFILE) -c $< -o $@

.PHONY:
	clean run prof

clean:
	rm $(BUILDDIR)/* $(MAIN) *.out 

run: $(MAIN)
	./adj ../data/file.txt

test: $(TESTDIR)/test.cc $(OBJS)
	$(CXX) $^ $(LIBS) $(CXXFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@
	./test

linalg_test: $(TESTDIR)/linalg_test.cu $(TESTOBJS)
	$(NVCXX) $^ $(LIBS) $(NVCCFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@
	./linalg_test


prof: PROFILE := -pg
prof: $(MAIN)
	./adj -e 10000000 -n 1000000 -k 1000
	gprof -b ./adj > tests/serial_profile
