#include "lanczos.h"
#include "adjMatrix.h"

void lanczosDecomp::decompose(const adjMatrix &A)
{
        unsigned n{A.get_n()}, offset{0u}, i{0u};
        std::vector<double> v(n);
        std::vector<std::vector<double>> Q_s(2, std::vector<double>(n));

        double x_norm = norm(x);

        std::copy(x.begin(), x.end(), Q_s[i].begin());
        std::for_each(Q_s[i].begin(), Q_s[i].end(), [=](double &a)
                      { a /= x_norm; });

        for (auto j = 0u; j < krylov_dim; j++)
        {

                // v = A*Q(:,j)
                sparse_adj_mat_vec_mult(A, Q_s[i], v);

                // alpha = v*Q(:,j)
                alpha[j] = std::inner_product(v.begin(), v.end(), Q_s[i].begin(), 0.0);

                offset = 0u;

                // v = v - alpha*Q(:,j)
                std::for_each(v.begin(), v.end(), [&](double &a)
                              { a -= alpha[j] * Q_s[i][offset++]; });

                if (j > 0)
                {
                        offset = 0u;
                        // v = v - beta*Q(:,j-1)
                        std::for_each(v.begin(), v.end(), [&](double &a)
                                      { a -= beta[j - 1] * Q_s[1 - i][offset++]; });
                }

                if (j < krylov_dim - 1)
                {
                        beta[j] = norm(v);
                        std::copy(v.begin(), v.end(), Q_s[1 - i].begin());
                        std::for_each(Q_s[1 - i].begin(), Q_s[1 - i].end(), [=](double &a)
                                      { a /= beta[j]; });
                }

                // Copying the Q_s column into the Q matrix (implemented as a 1d row maj vector)
                offset = 0u;
                std::for_each(Q_s[i].begin(), Q_s[i].end(), [&](double &a) { Q[j + offset++ * krylov_dim] = a; });
                i = 1-i;
        }
}

std::ostream &operator<<(std::ostream &os, const lanczosDecomp &D)
{
        os << "\nAlpha: \n";
        for (auto it = D.alpha.begin(); it != D.alpha.end(); it++)
                os << *it << " ";

        os << "\n\nBeta: \n";
        for (auto it = D.beta.begin(); it != D.beta.end(); it++)
                os << *it << " ";

        auto offset{0u};
        os << "\n\nQ:\n";
        for (auto it = D.Q.begin(); it != D.Q.end(); it++)
        {
                os << *it << "\t";
                if (offset++ % D.krylov_dim == D.krylov_dim - 1)
                        os << '\n';
        }

        os << '\n';
        return os;
}

double lanczosDecomp::norm(const std::vector<double> &v)
{
        double norm{0.0};
        for (auto it = v.begin(); it != v.end(); it++)
                norm += (*it) * (*it);
        return std::sqrt(norm);
}

void lanczosDecomp::get_ans() const
{
        std::cout << "Answer vector:\n";
        for (auto it = ans.begin(); it != ans.end(); ++it)
        {
                std::cout << *it << '\n';
        }
}