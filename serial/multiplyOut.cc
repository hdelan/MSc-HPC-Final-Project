#include "eigen.h"
#include "lanczos.h"
#include "lapacke.h"
#include "cblas.h"

void print_matrix(unsigned rows, unsigned cols, double * A) {
        std::cout << "Printing matrix for "<<rows<<" rows and " << cols<< "cols\n";
        for (auto i=0u; i<rows; ++i) {
                for (auto j=0u; j<cols; ++j){
                        std::cout << A[i*cols+j] << " ";
                }
                std::cout << '\n';
        }
}

void exp_func(double & a) {
        a = std::exp(a);
}

void multOut(lanczosDecomp & D, eigenDecomp & E, adjMatrix & A) {

        auto k {E.eigenvalues.size()};
        auto n {A.get_n()};
        
        // Applying function
        std::for_each(E.eigenvalues.begin(), E.eigenvalues.end(), exp_func);
        
        //print_matrix(3, 1, &E.eigenvalues[0]);
        
        // Elementwise multiplying of f(lambda) by first row of eigenvectors
        auto offset {0u};
        std::for_each(E.eigenvalues.begin(), E.eigenvalues.end(), [&](double & a) {a *= E.eigenvectors.at(offset++)*D.x_norm;});

        //print_matrix(3, 1, &E.eigenvalues[0]);
        
        double * Qa {new double [n*k]};
        double * Va {new double [k*k]};
        double * QVa {new double [n*k]};

        for (auto i=0u; i<k; ++i) {
                for (auto j=0u; j<k; ++j) {
                        //Q[j+i*k] = eigenvectors.at(i).at(j); // Col major
                        Va[j+i*k] = E.eigenvectors.at(i*k+j); // Row major
                }
        }
        
        for (auto i=0u; i<k; ++i) {
                for (auto j=0u; j<n; ++j) {
                        //Q[j+i*k] = D.Q.at(i).at(j); // Col major
                        Qa[j*k+i] = D.Q.at(i).at(j); // Row major
                }
        }
        
/*
        std::cout << "\nQ:\n";
        print_matrix(n, k, Qa);
        std::cout << "\nV:\n";
        print_matrix(k, k, Va);

*/
        cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, n, k, k, 1, Qa, k, Va, k, 1, QVa, k);
 
        cblas_dgemv(CblasRowMajor, CblasNoTrans, n, k, 1, QVa, k, &E.eigenvalues[0], 1, 1, &A.ans[0],1);

        delete[](Qa);
        delete[](Va);
        delete[](QVa);
        
}
