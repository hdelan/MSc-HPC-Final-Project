#ifndef CU_MULT_OUT12435
#define CU_MULT_OUT12435

#include "cu_lanczos.h"
#include "eigen.h"
#include "adjMatrix.h"

template <typename T>
void cu_multOut(lanczosDecomp<T> & L, eigenDecomp<T> & E, adjMatrix & A);

#endif
