SHELL 		:= /bin/bash
CXX 		:= g++
CXXFLAGS 	:= -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS 	 	:= -lblas -llapack -llapacke -lm #"-I/usr/local/opt/lapack/include"

MAIN 		:= adj

SRCDIR 		:= ./lib
BUILDDIR 	:= ./build

FILES 		:= adjMatrix.cc \
       			lanczos.cc \
       			eigen.cc \
       			multiplyOut.cc \
       			sparse_mult.cc \
       			helpers.cc


OBJS		:= $(FILES:%.cc=$(BUILDDIR)/%.o)

PROFILE 	:= 

########################################################################

$(MAIN): main.cc $(OBJS) 
	$(CXX) $^ $(LIBS) $(CXXFLAGS) $(LIBS) $(PROFILE) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc $(SRCDIR)/%.h
	$(CXX) $(CXXFLAGS) $(PROFILE) -c $< -o $@

.PHONY:
	clean run prof

clean:
	rm $(BUILDDIR)/* $(MAIN) *.out

run: $(MAIN)
	./adj ../data/file.txt 

prof: PROFILE := -pg
prof: $(MAIN)
	./adj -f ../data/California.mtx -k 1000
	gprof ./adj > serial_profile
