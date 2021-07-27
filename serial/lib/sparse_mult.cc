#include "sparse_mult.h"


/* --------------------------------------------------------------------------*/
/**
 * \brief:       This is only suitable for A, an adjacency matrix with entries 0 or 1
 *
 * \param:       x
 */
/* ----------------------------------------------------------------------------*/
template <typename T>
void sparse_adj_mat_vec_mult(const adjMatrix & A, const T * const in, T * const out) {
        for (auto i = 0u; i < A.n; ++i) {
                out[i] = 0.0;
        }

        for (auto i = 0u; i < 2*A.edge_count; ++i) {
                out[A.row_idx[i]] += in[A.col_idx[i]]; 
        }
}
 
template void sparse_adj_mat_vec_mult(const adjMatrix & A, const double * in, double * const out);
template void sparse_adj_mat_vec_mult(const adjMatrix & A, const float * in, float * const out);
