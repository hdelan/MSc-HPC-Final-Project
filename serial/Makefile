
SHELL = /bin/bash
CXX = g++
CXXFLAGS = -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS = 
MAIN = adj

########################################################################

$(MAIN): $(MAIN).o
	$(CXX) -o $@ $< $(LIBS) $(CXXFLAGS)

$(MAIN).o: main.cc
	$(CXX) -c $< $(LIBS) $(CXXFLAGS)



clean:
	rm *.o
