
SHELL = /bin/bash
CXX = g++
CXXFLAGS = -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS = 
LAPACKE = -lblas -llapack -llapacke -lm #"-I/usr/local/opt/lapack/include"
MAIN = adj
OBJECTS = adjMatrix.o main.o sparse_mult.o lanczos.o eigen.o multiplyOut.o helpers.o

########################################################################

$(MAIN): $(OBJECTS)
	$(CXX) -o $@ $^ $(LIBS) $(CXXFLAGS) $(LAPACKE)

adjMatrix.o: adjMatrix.cc adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

sparse_mult.o: sparse_mult.cc adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

main.o: main.cc adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS) $(LAPACKE)

lanczos.o: lanczos.cc lanczos.h adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

eigen.o: eigen.cc eigen.h adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS) $(LAPACKE)

multiplyOut.o: multiplyOut.cc eigen.h lanczos.h adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS) $(LAPACKE)

helpers.o: helpers.cc
	$(CXX) -c $< $(LIBS) $(CXXFLAGS) $(LAPACKE)


.PHONY:
	clean run

clean:
	rm *.o $(MAIN)

run: $(MAIN)
	./adj ../data/file.txt
