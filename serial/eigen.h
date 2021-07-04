#ifndef EIGEN_H_732234189
#define EIGEN_H_732234189

#include "lanczos.h"
#include "lapacke.h"
#include <vector>

class eigenDecomp {
        public: 
                eigenDecomp() = delete;
                eigenDecomp(lanczosDecomp & D) : 
                        eigenvalues(D.alpha.begin(), D.alpha.end()),
                        eigenvectors(D.krylov_dim*D.krylov_dim)
        {
                decompose(D);
        };
                eigenDecomp(eigenDecomp &)=delete;
                eigenDecomp & operator=(eigenDecomp &)=delete;
                ~eigenDecomp()=default;

        private:
                std::vector<double> eigenvalues;
                std::vector<double> eigenvectors;

                void decompose(lanczosDecomp & D);
};
#endif
