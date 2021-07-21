#include "lanczos.h"
#include "adjMatrix.h"
#include "sparse_mult.h"

void lanczosDecomp::decompose(const adjMatrix &A)
{
        unsigned n{A.get_n()}, i{0u};
        double * v {new double[n]};
        double * Q_raw(new double [2*n]);
        double ** Q_s (new double * [2]);
        Q_s[0] = Q_raw;
        Q_s[1] = &Q_raw[n];

        double x_norm = norm(x);

        for (auto k=0u;k<n;k++) 
                Q_s[i][k] = x[k]/x_norm;

        for (auto j = 0u; j < krylov_dim; j++)
        {

                // v = A*Q(:,j)
                sparse_adj_mat_vec_mult(A, Q_s[i], v);

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
        }
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

double lanczosDecomp::inner_prod(const double * const v, const double * const w, const unsigned N) const {
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