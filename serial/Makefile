
SHELL = /bin/bash
CXX = g++
CXXFLAGS = -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS = 
MAIN = adj
OBJECTS = adjMatrix.o main.o sparse_mult.o lanczos.o

########################################################################

$(MAIN): $(OBJECTS)
	$(CXX) -o $@ $^ $(LIBS) $(CXXFLAGS)

adjMatrix.o: adjMatrix.cc adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

sparse_mult.o: sparse_mult.cc adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

main.o: main.cc adjMatrix.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

lanczos.o: lanczos.cc lanczos.h
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)

.PHONY:
	clean run

clean:
	rm *.o

run: $(MAIN)
	./adj file.txt
