#include "lib/adjMatrix.h"
#include "lib/eigen.h"
#include "lib/lanczos.h"
#include "lib/helpers.h"
#include "lib/SPMV.h"
#include "lib/multiplyOut.h"
#include "lib/cu_linalg.h"
#include "lib/cu_SPMV.h"

#include <iomanip>

int main(int argc, char *argv[])
{

        std::string filename{"../data/file.txt"};
        long unsigned krylov_dim{1};

        long unsigned n{100}, edges{40};
        long unsigned deg {0};
        bool verbose {true};

        unsigned width {40}; // for formatting text

        assert(n > 0 && edges > 0 && deg < n);

        parseArguments(argc, argv, filename, krylov_dim, verbose, n, deg, edges);
        
        adjMatrix A;

        char make_matrix{'f'};

        switch (make_matrix)
        {
        case 'f': // Read matrix from file
        {
                std::ifstream fs;
                fs.open(filename);
                assert (!fs.fail() && "File opening failed\n");

                fs >> n >> n >> edges;

                adjMatrix B(n, edges, fs);
                fs.close();
                A = std::move(B);
                break;
        }
        case 'r':  // Generate random matrix
        {
                adjMatrix B(n, edges);
                A = std::move(B);
                break;
        }
        case 's': // Generate 
        {
                adjMatrix B(n, 's');
                A = std::move(B);
                break;
        }
        case 'b': // generate random Barabasi matrix
        {
                adjMatrix B(n, deg, 'b');
                A = std::move(B);
                break;

        }
        }
        krylov_dim = std::min(n-1, krylov_dim);

        edges = A.get_edges();

        std::cout << std::setw(width) << std::setfill('~') << '\n' << std::setfill(' ');

        std::cout << "n: " << A.get_n() << '\n';
        std::cout << "edges: " << A.get_edges() << '\n';
        std::cout << "krylov dimension: " << krylov_dim << '\n';
        std::cout << std::setw(width) << std::setfill('~') << '\n' << std::setfill(' ');

        //std::cout << "A: \n" << A << '\n';
        //A.print_full();
        std::vector<double> x(n, 1);

        //assert(krylov_dim <= n);

        lanczosDecomp L(A, krylov_dim, &x[0]);
        
        /*
        //std::cout << L;
        eigenDecomp E(L);
        //std::cout << E;
        multOut(L, E, A);

        std::cout << '\n';

        //if (verbose) L.get_ans();
        std::cout << '\n';
        */
        return 0;
}
