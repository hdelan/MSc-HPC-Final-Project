SELL 		:= /bin/bash
CXX 		:= g++
CXXFLAGS 	:= -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS 	 	:= -lblas -llapack -llapacke -lm #"-I/usr/local/opt/lapack/include"
OPT		:= -O3

MAIN 		:= adj

SRCDIR 		:= ./lib
BUILDDIR 	:= ./build

FILES 		:= adjMatrix.cc \
       			lanczos.cc \
       			eigen.cc \
       			multiplyOut.cc \
       			SPMV.cc \
       			helpers.cc \
			make_graph.cc

OBJS		:= $(FILES:%.cc=$(BUILDDIR)/%.o)

PROFILE 	:= 

########################################################################

$(MAIN): main.cc $(OBJS) 
	$(CXX) $^ $(LIBS) $(CXXFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc #$(SRCDIR)/%.h
	$(CXX) $(CXXFLAGS) $(OPT) $(PROFILE) -c $< -o $@

.PHONY:
	clean run prof

clean:
	rm $(BUILDDIR)/*.o $(MAIN) 

run: $(MAIN)
	./adj -f ../data/generated/file.txt

numerical_test: tests/numerical_test.cc $(OBJS)
	$(CXX) $^ $(LIBS) $(CXXFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@
	./numerical_test

numerical_test_orthog: tests/numerical_test_orthog.cc $(OBJS)
	$(CXX) $^ $(LIBS) $(CXXFLAGS) $(OPT) $(LIBS) $(PROFILE) -o $@
	./numerical_test_orthog

