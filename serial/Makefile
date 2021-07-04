
SHELL = /bin/bash
CXX = g++
CXXFLAGS = -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS = 
LAPACKE = "-I/usr/local/opt/lapack/include"
MAIN = adj
OBJECTS = adjMatrix.o main.o sparse_mult.o lanczos.o eigen.o

########################################################################

$(MAIN): $(OBJECTS)
	$(CXX) -o $@ $^ $(LIBS) $(CXXFLAGS) $(LAPACKE)

adjMatrix.o: adjMatrix.cc adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

sparse_mult.o: sparse_mult.cc adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

main.o: main.cc adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS) $(LAPACKE)

lanczos.o: lanczos.cc lanczos.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

eigen.o: eigen.cc eigen.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS) $(LAPACKE)

.PHONY:
	clean run

clean:
	rm *.o

run: $(MAIN)
	./adj file.txt
