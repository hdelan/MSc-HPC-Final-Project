#ifndef MULTIPLY_H_234234
#define MULTIPLY_H_234234

#include "adjMatrix.h"
#include "eigen.h"
#include "lanczos.h"

void print_matrix(unsigned rows, unsigned cols, double * A);
void multOut(lanczosDecomp & D, eigenDecomp & E, adjMatrix & A);

#endif
