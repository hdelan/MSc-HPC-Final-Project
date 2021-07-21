#ifndef MULTIPLY_H_234234
#define MULTIPLY_H_234234

#include "adjMatrix.h"
#include "eigen.h"
#include "lanczos.h"

void print_matrix(unsigned, unsigned, double *);
void multOut(lanczosDecomp &, eigenDecomp &, adjMatrix &);

#endif
