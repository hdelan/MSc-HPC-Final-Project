#include "lanczos.h"
#include "adjMatrix.h"
#include "sparse_mult.h"

void lanczosDecomp::decompose(const adjMatrix &A)
{
        long unsigned n{A.get_n()}, i{0u};
        double * v {new double[n]};
        double * Q_raw(new double [2*n]);
        double * Q_s[2] {Q_raw, &Q_raw[n]};     // Tmp contiguous columns to use before storing
                                                // in row-major matrix

        double x_norm = norm(x);

        for (auto k=0u;k<n;k++) 
                Q_s[i][k] = x[k]/x_norm;

        std::cout << "x:\n";
        for (auto j=0u;j<5;j++) std::cout << Q_s[i][j] << " ";
        std::cout << '\n';

        for (auto j = 0u; j < krylov_dim; j++)
        {

                // v = A*Q(:,j)
                sparse_adj_mat_vec_mult(A, Q_s[i], v);

                std::cout << v[0] << " " << v[1] << '\n';

                // alpha = v*Q(:,j)
                alpha[j] = inner_prod(v, Q_s[i], n);

                // v = v - alpha*Q(:,j)
                for (auto k=0u;k<n;k++) 
                        v[k] -= alpha[j]*Q_s[i][k];

                if (j > 0)
                {
                        // v = v - beta*Q(:,j-1)
                        for (auto k=0u;k<n;k++) 
                                v[k] -= beta[j-1]*Q_s[1-i][k];
                }

                if (j < krylov_dim - 1)
                {
                        beta[j] = norm(v);
                        for (auto k=0u;k<n;k++) 
                                Q_s[1-i][k] = v[k]/beta[j];
                }

                // Copying the Q_s column into the Q matrix (implemented as a 1d row maj vector)
                for (auto k=0u;k<n;k++) 
                        Q[j+k*krylov_dim] = Q_s[i][k];

                i = 1-i;
/*
        */
        }
        std::cout << "\nAlpha: ";
        for (auto j=0u;j<krylov_dim;j++) std::cout << alpha[j] << " ";
        std::cout << "\nBeta: ";
        for (auto j=0u;j<krylov_dim-1;j++) std::cout << beta[j] << " ";
        std::cout << '\n';
        delete[] v; delete[] Q_raw;
}

std::ostream &operator<<(std::ostream &os, const lanczosDecomp &D)
{
        os << "\nAlpha: \n";
        for (auto i=0u;i<D.n;i++)
                os << D.alpha[i] << " ";

        os << "\n\nBeta: \n";
        for (auto i=0u;i<D.n-1;i++)
                os << D.beta[i] << " ";

        os << "\n\nQ:\n";
        for (auto i=0u;i<D.n-1;i++){
                os << D.Q[i] << " ";
                if (i%D.krylov_dim == D.krylov_dim-1) os << '\n';
        }

        os << '\n';
        return os;
}

double lanczosDecomp::norm(const double * v) const
{
        double norm{0.0};
        for (auto i=0u;i<n;i++)
                norm += v[i] * v[i];
        return std::sqrt(norm);
}

double lanczosDecomp::inner_prod(const double * const v, const double * const w, const long unsigned N) const {
        double ans {0.0};
        for (auto i=0u;i<N;i++) {
                ans += v[i]*w[i];
        }
        return ans;
}

void lanczosDecomp::get_ans() const
{
        std::cout << "Answer vector:\n";
        
        for (auto i=0u;i<n;i++) 
                std::cout << ans[i] << '\n';
}


void lanczosDecomp::check_ans(const double * analytic_ans) const {
        std::vector<double> diff(n);
        for (auto i=0u;i<n;i++) {
                diff[i] = std::abs(ans[i] - analytic_ans[i]);
        }
        auto max_it = std::max_element(diff.begin(), diff.end());
        auto max_idx = std::distance(diff.begin(), max_it);
        std::cout << "\nMax difference of " << *max_it
                << " found at index\nlanczos[" << max_idx << "] \t\t= " << ans[max_idx]
                << "\nanalytic_ans["<<max_idx<<"] \t= "<<analytic_ans[max_idx]<<'\n';

        std::cout << "Total norm of differences: " << norm(&diff[0]) << '\n'; 
}