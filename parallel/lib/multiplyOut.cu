#include "multiplyOut.h"
#include "lapacke.h"
#include "cblas.h"
#include <iomanip>

void naive_dgemm(double * a, double * b, const unsigned rows, const unsigned cols, double * ans);
void naive_dgemv(double * a, double * v, const unsigned rows, const unsigned cols, double * ans);

void exp_func(double & a) {
        a = std::exp(a);
}

void multOut(lanczosDecomp & L, eigenDecomp & E, adjMatrix & A) {

        auto n {L.get_n()};
        
        // Applying function
        for (auto j=0u;j<L.krylov_dim;j++) exp_func(E.eigenvalues[j]);
        
        // Elementwise multiplying of f(lambda) by first row of eigenvectors
        for (auto j=0u;j<L.krylov_dim;j++) E.eigenvalues[j] *= L.x_norm * E.eigenvectors[j];

        //print_matrix(3, 1, &E.eigenvalues[0]);
        
        double * QV {new double [n*L.krylov_dim]};
        /*
        for (auto j = 0u; j < n; j++)
        {
                for (auto k = 0u; k < L.krylov_dim; k++)
                        std::cout << std::setprecision(20) << Q[k + j * L.krylov_dim] << " ";
                std::cout << '\n';
        }
        for (auto j = 0u; j < L.krylov_dim; j++)
        {
                for (auto k = 0u; k < L.krylov_dim; k++)
                        std::cout << std::setprecision(20) << E.eigenvectors[k + j * L.krylov_dim] << " ";
                std::cout << '\n';
        }
        */
/*
        // Getting QV (n x k)
        for (auto i=0u;i<n;i++) {
                for (auto j=0u;j<L.krylov_dim;j++) {
                        QV[i*L.krylov_dim+j] = 0.0;
                        for (auto k=0u;k<L.krylov_dim;k++) {
                                QV[i*L.krylov_dim+j] += L.Q[i*L.krylov_dim+k]*E.eigenvectors[k*L.krylov_dim+j];
                        }
                }
        }
        */
        // This call to cblas_dgemm was not working for me!
        auto k = L.get_krylov();
        //naive_dgemm(L.Q, E.eigenvectors, n,k, QV);
        cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, k, k, 1, L.Q, k, E.eigenvectors, k, 0, QV, k);
// PRINT OUT QV
/*
        std::cout << "\nQV\n";
        for (auto j = 0u; j < L.krylov_dim; j++)
        {
                for (auto k = 0u; k < L.krylov_dim; k++)
                        std::cout <<std::setprecision(4)<< QV[k + j * L.krylov_dim] << " ";
                std::cout << '\n';
        }
        */
/*
        */
        // Getting QV*f(lambda)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, L.krylov_dim, 1, QV, k, &E.eigenvalues[0], 1, 0, &L.ans[0],1);
        //naive_dgemv(QV, &E.eigenvalues[0], n, L.krylov_dim, &L.ans[0]);

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

void naive_dgemm(double * a, double * b, const unsigned rows, const unsigned cols, double * ans) {
  for (int i=0;i<rows;i++) {
    for (int j=0;j<cols;j++) {
      ans[i*cols+j] = 0.0;
      for (int k=0;k<cols;k++) {
        ans[i*cols+j] += a[i*cols+k]*b[k*cols+j];
      }
    }
  }
}

void naive_dgemv(double * a, double * v, const unsigned rows, const unsigned cols, double * ans) {
  for (int i=0;i<rows;i++) {
    ans[i] = 0.0;
    for (int j=0;j<cols;j++) 
      ans[i] += a[i*cols+j]*v[j];
  }
}
