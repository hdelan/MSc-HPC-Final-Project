#include "multiplyOut.h"
#include "lapacke.h"
#include "cblas.h"

void exp_func(double & a) {
        a = std::exp(a);
}

void multOut(lanczosDecomp & L, eigenDecomp & E, adjMatrix & A) {

        auto k {E.k};
        auto n {E.n};
        
        // Applying function
        for (auto j=0u;j<k;j++) exp_func(E.eigenvalues[j]);
        
        // Elementwise multiplying of f(lambda) by first row of eigenvectors
        for (auto j=0u;j<k;j++) E.eigenvalues[j] *= L.x_norm * E.eigenvectors[j];

        //print_matrix(3, 1, &E.eigenvalues[0]);
        
        double * QV {new double [n*k]};

        // Getting QV (n x k)
        cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, k, k, 1, &L.Q[0], k, E.eigenvectors, k, 1, QV, k);
        
        // Getting QV*f(lambda)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, k, 1, QV, k, &E.eigenvalues[0], 1, 1, &L.ans[0],1);

        delete[](QV);
        
}

void print_matrix(unsigned rows, unsigned cols, double * A) {
        std::cout << "Printing matrix for "<<rows<<" rows and " << cols<< "cols\n";
        for (auto i=0u; i<rows; ++i) {
                for (auto j=0u; j<cols; ++j){
                        std::cout << A[i*cols+j] << " ";
                }
                std::cout << '\n';
        }
}
