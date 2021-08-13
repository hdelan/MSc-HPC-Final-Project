#include "SPMV.h"


/* --------------------------------------------------------------------------*/
/**
 * \brief:       This is only suitable for A, an adjacency matrix with entries 0 or 1
 *
 * \param:       x
 */
/* ----------------------------------------------------------------------------*/
template <typename T>
void spMV(const adjMatrix & A, const T * const in, T * const out) {
        for (auto i = 0u; i < A.n; ++i) {
                out[i] = 0.0;
        }

        for (auto i = 0u; i < A.n; ++i) {
                for (auto j=A.row_offset[i];j<A.row_offset[i+1];j++)
                        out[i] += in[A.col_idx[j]]; 
        }
}
 
template void spMV(const adjMatrix & A, const double * in, double * const out);
template void spMV(const adjMatrix & A, const float * in, float * const out);
