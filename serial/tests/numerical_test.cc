#include "../lib/adjMatrix.h"
#include "../lib/eigen.h"
#include "../lib/lanczos.h"
#include "../lib/helpers.h"
#include "../lib/SPMV.h"
#include "../lib/multiplyOut.h"

#include <iomanip>
#include <random>

int main(int argc, char *argv[])
{
        std::string dir{"../data/NotreDame_yeast/"};
        std::string filename{dir + "data.mtx"};
        bool verbose;
        
        long unsigned krylov_dim{25};

        unsigned max_eigen{100}, eigens{100};

        long unsigned n, edges, deg;

        unsigned width{77}; // for formatting text to std::cout
        
        parseArguments(argc, argv, filename, krylov_dim, verbose, n, deg, edges);

        std::cout << std::setw(width) << std::setfill('~') << '\n'
                  << std::setfill(' ');
        std::cout << "Comparing the Lanczos based approximation with an analytic answer, using the \nfact that: "
                  << "\n\n\tf(A)v = f(lambda)v\n\nfor an eigenpair (v,lambda).\n\n"
                  << "Using the first "<<eigens<<" eigenpairs to construct our problem vector x.\n\n";
        std::cout << std::setw(width) << std::setfill('~') << '\n'
                  << std::setfill(' ');

        std::ifstream fs;
        fs.open(filename);
        //assert (!fs.fail());

        fs >> n >> n >> edges;

        assert(n > 0 && edges > 0);

        adjMatrix A(n, edges, fs);
        fs.close();

        fs.open(dir + "eigvecs.csv");
        std::vector<std::vector<double>> eigvecs(max_eigen, std::vector<double>(n));
        for (auto j = 0u; j < n; j++)
                for (auto i = 0u; i < max_eigen; i++)
                        fs >> eigvecs[i][j];
        fs.close();
        
        fs.open(dir + "eigvals.csv");
        std::vector<double> eigvals(max_eigen);
        for (auto i = 0u; i < max_eigen * max_eigen; i++)
        {
                double tmp;
                fs >> tmp;
                if (i % max_eigen == i / max_eigen)
                        eigvals[i % max_eigen] = tmp;
        }
        fs.close();

        unsigned seed{1234};
        std::mt19937 gen{seed};
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<double> coeff(eigens);

        for (auto i = 0u; i < eigens; i++)
                coeff[i] = U(gen);

        std::vector<double> x(n);

        for (auto i = 0u; i < n; i++)
        {
                x[i] = 0.0;
                for (auto j = 0u; j < eigens; j++)
                        x[i] += coeff[j] * eigvecs[j][i];
        }
        edges = A.get_edges();
        n = A.get_n();

        std::cout << "n: " << A.get_n() << '\n';
        std::cout << "edges: " << A.get_edges() << '\n';
        std::cout << "krylov dimension: " << krylov_dim << '\n';
        std::cout << std::setw(width) << std::setfill('~') << '\n'
                  << std::setfill(' ');

        {
                lanczosDecomp L(A, krylov_dim, &x[0]);
                eigenDecomp E(L);
                multOut(L, E, A);

                // Getting the analytic answer since exp(A)v=exp(lambda)v when v is an
                // eigenvector and eigenvectors are orthogonal
                for (auto i = 0u; i < eigens; i++)
                        coeff[i] *= std::exp(eigvals[i]);

                for (auto i = 0u; i < n; i++)
                {
                        x[i] = 0.0;
                        for (auto j = 0u; j < eigens; j++)
                                x[i] += coeff[j] * eigvecs[j][i];
                }

                L.check_ans(&x[0]);
                //L.get_ans();
        }

        std::cout << '\n';
        std::cout << '\n';

        return 0;
}

        
