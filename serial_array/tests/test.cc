#include "../lib/adjMatrix.h"
#include "../lib/eigen.h"
#include "../lib/lanczos.h"
#include "../lib/helpers.h"
#include "../lib/sparse_mult.h"
#include "../lib/multiplyOut.h"
#include "lapacke.h"

int main(int argc, char *argv[])
{
        std::string dir{"../data/NotreDame_yeast/"};
        std::string filename{dir+"data.mtx"};
        unsigned krylov_dim{4};

        long unsigned n, edges;

        std::ifstream fs;
        fs.open(filename);
        //assert (!fs.fail());

        fs >> n >> n >> edges;

        std::cout << n << " " << edges << std::endl;
        
        assert(n > 0 && edges > 0);

        adjMatrix A(n, edges, fs);
        fs.close();

        fs.open(dir + "eigvecs.csv");
        std::vector<std::vector<double>> eigvecs(6, std::vector<double>(n));
        for (auto j = 0u; j < n; j++)
                for (auto i = 0u; i < 6; i++)
                        fs >> eigvecs[i][j];
        fs.close();
/*
                std::cout << "Eigvecs:\n";
        for (auto i=0u;i<5;i++) {
                for (auto j=0u;j<6;j++) 
                        std::cout << eigvecs[j][i] << " ";
                std::cout << '\n';
        }
*/
        fs.open(dir + "eigvals.csv");
        double *eigvals{new double[6]};
        for (auto i = 0u; i < 36; i++)
        {
                double tmp;
                fs >> tmp;
                if (i % 6 == i / 6)
                        eigvals[i%6] = tmp;
        }
        fs.close();

        std::cout << "Eigenvalues: \n";
        for (int i=0u;i<6;i++) std::cout << eigvals[i] << " ";
        std::cout << "\n";

        edges = A.get_edges();
        n = A.get_n();

        std::cout << "n: " << A.get_n() << '\n';
        std::cout << "edges: " << A.get_edges() << '\n';
        std::cout << "krylov dimension: " << krylov_dim << '\n';

        /*
        std::cout << "A: \n"
                  << A << '\n';
        A.print_full();
        */

       for (auto i=0u;i<6;i++) {
                //std::vector<double> x(eigvecs[i].begin(), eigvecs[i].end());
                std::vector<double> x(n, 1);
                lanczosDecomp L(A, krylov_dim, &x[0]);
                eigenDecomp E(L);
                multOut(L, E, A);
                
                // Getting the analytic answer since exp(A)v=exp(lambda)v when v is an
                // eigenvector and eigenvectors are orthogonal
                std::for_each(x.begin(), x.end(), [&](double & a) { a *= std::exp(eigvals[i]);});
                L.check_ans(&x[0]);

       }


        //assert(krylov_dim <= n);

        //std::cout << L;

        //std::cout << E;

        std::cout << '\n';
        //L.get_ans();
        std::cout << '\n';

        //delete[] raw_adj_matrix;

        return 0;
}
