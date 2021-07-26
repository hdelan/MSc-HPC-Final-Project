#include "eigen.h"

void eigenDecomp::decompose(lanczosDecomp & D) {
        LAPACKE_dstevd(LAPACK_ROW_MAJOR, 'V', D.krylov_dim, &eigenvalues[0], &D.beta[0], &eigenvectors[0], D.krylov_dim);
}
/*
                LAPACKE_dstevd	(	int 	matrix_layout,  LAPACKE_ROW_MAJOR
char 	jobz,
lapack_int 	n,
double * 	d,
double * 	e,
double * 	z,
lapack_int 	ldz 
)*/
