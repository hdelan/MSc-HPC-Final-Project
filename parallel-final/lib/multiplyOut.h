#ifndef MULTIPLY_H_234234
#define MULTIPLY_H_234234

#include "adjMatrix.h"
#include "eigen.h"
#include "cu_lanczos.h"

#include "cublas_v2.h"
#include "cblas.h"

template <typename T>
void multOut(lanczosDecomp<T> &, eigenDecomp<T> &, adjMatrix &, bool Qtrans);

#endif

