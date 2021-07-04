#include "lanczos.h"
#include "adjMatrix.h"

void lanczosDecomp::decompose(const adjMatrix & A) {
        unsigned n {A.get_n()}, offset {0u};
        std::vector<double> v (n);

        auto norm = [](const auto & v) {
                return sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0));
        };
        
        auto print_a = [](const auto & a) {
                std::cout << a << " ";
        };

        double x_norm = norm(x);
        std::cout << "x norm: " << x_norm << std::endl;

        std::copy(x.begin(), x.end(), Q[0].begin());
        std::for_each(Q[0].begin(), Q[0].end(), print_a);
        std::cout << '\n';
        std::for_each(Q[0].begin(), Q[0].end(), [=](double & a) {a /= x_norm;});
        std::for_each(Q[0].begin(), Q[0].end(), print_a);
        std::cout << '\n';

        for (auto j = 0u; j < krylov_dim; j++) {
                std::cout << "\nj:" << j << '\n';
                sparse_adj_mat_vec_mult(A, Q.at(j), v);
                
                std::cout << "\nQ(:, "<<j<<"): \n";
                std::for_each(Q.at(j).begin(), Q.at(j).end(), print_a);
                std::cout << '\n';
                std::cout << "\nv: \n";
                std::for_each(v.begin(), v.end(), print_a);
                std::cout << '\n';

                alpha.at(j) = std::inner_product(v.begin(), v.end(), Q.at(j).begin(), 0.0);
                std::cout << "Inner product of Q(:, "<< j <<") and v: \n" << alpha.at(j) << '\n';
                offset = 0u;
                std::for_each(v.begin(), v.end(), [&](double & a) {a -= alpha.at(j)*Q.at(j).at(offset++);});
                std::cout << "\nv-alpha*Q(:,"<<j<<"): \n";
                std::for_each(v.begin(), v.end(), print_a);
                std::cout << '\n';
                if (j > 0) {
                        offset = 0u;
                        std::for_each(v.begin(), v.end(), [&](double & a) {a -= beta.at(j-1)*Q.at(j-1).at(offset++);});
                }
                if (j < krylov_dim-1) {
                        beta.at(j) = norm(v);
                        std::cout << "Beta("<<j<<"): "<< beta.at(j) << '\n';
                        std::copy(v.begin(), v.end(), Q.at(j+1).begin());
                        std::for_each(Q.at(j+1).begin(), Q.at(j+1).end(), [=](double & a) {a /= beta.at(j);});
                }
        }
}

