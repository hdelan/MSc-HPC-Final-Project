#include "lanczos.h"
#include "adjMatrix.h"

void print_a(const double & a) {
        std::cout << a << " ";
}

void lanczosDecomp::decompose(const adjMatrix & A) {
        unsigned n {A.get_n()}, offset {0u};
        std::vector<double> v (n);

        /*
           auto norm = [](const auto & v) {
           return sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0));
           };
           */


        double x_norm = norm(x);

        std::copy(x.begin(), x.end(), Q[0].begin());
        std::for_each(Q[0].begin(), Q[0].end(), [=](double & a) {a /= x_norm;});

        for (auto j = 0u; j < krylov_dim; j++) {
                sparse_adj_mat_vec_mult(A, Q.at(j), v);

                alpha.at(j) = std::inner_product(v.begin(), v.end(), Q.at(j).begin(), 0.0);
                
                offset = 0u;
                std::for_each(v.begin(), v.end(), [&](double & a) {a -= alpha.at(j)*Q.at(j).at(offset++);});
                if (j > 0) {
                        offset = 0u;
                        std::for_each(v.begin(), v.end(), [&](double & a) {a -= beta.at(j-1)*Q.at(j-1).at(offset++);});
                }
                if (j < krylov_dim-1) {

                        beta.at(j) = norm(v);
                        std::copy(v.begin(), v.end(), Q.at(j+1).begin());
                        std::for_each(Q.at(j+1).begin(), Q.at(j+1).end(), [=](double & a) {a /= beta.at(j);});
                }
        }
}

