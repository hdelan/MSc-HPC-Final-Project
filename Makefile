
SHELL = /bin/bash
CXX = g++
CXXFLAGS = -Wall -Wextra -Wno-unused-parameter -std=c++17
LIBS = 
LAPACKE = -lblas -llapack -llapacke -lm #"-I/usr/local/opt/lapack/include"
MAIN = adj

SRCDIR = ./serial/lib
BUILDDIR = ./build

FILES = adjMatrix.cc \
       sparse_mult.cc \
       lanczos.cc \
       eigen.cc \
       multiplyOut.cc \
       helpers.cc

SRCS = $(FILES:%=$(SRCDIR)/%)
OBJS = $(FILES:%.cc=$(BUILDDIR)/%.o)

INCLUDES = eigen.h lanczos.h adjMatrix.h

PROFILE = -pg
########################################################################

all:
	echo $(OBJS)
	echo $(SRCS)

.PHONY:
	clean run

clean:
	rm $(OBJDIR)*.o $(MAIN)

profile: main
	prof = true
	

run: $(MAIN)
	./adj ../data/file.txt
