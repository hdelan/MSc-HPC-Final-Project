#ifndef SPARSE_MULT_H_82384
#define SPARSE_MULT_H_82384

#include "adjMatrix.h"

template <typename T>
void sparse_adj_mat_vec_mult(const adjMatrix &, const std::vector<T>, std::vector<T> &);

#endif
