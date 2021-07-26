#include "lib/adjMatrix.h"
#include "lib/eigen.h"
#include "lib/lanczos.h"
#include "lib/helpers.h"
#include "lib/sparse_mult.h"
#include "lib/multiplyOut.h"

int main(int argc, char *argv[])
{

        std::string filename{"../data/file.txt"};
        long unsigned krylov_dim{10};

        long unsigned n{100}, edges{40};
        unsigned deg {1};

        assert(n > 0 && edges > 0 && deg > 0);

        parseArguments(argc, argv, filename, krylov_dim);
        adjMatrix A;

        char make_matrix{'b'};
        if (filename != "") make_matrix = 'f';


        switch (make_matrix)
        {
        case 'f':
        {
                std::ifstream fs;
                fs.open(filename);
                //assert (!fs.fail());

                fs >> n >> n >> edges;

                adjMatrix B(n, edges, fs);
                fs.close();
                A = std::move(B);
                break;
        }
        case 'r':
        {
                adjMatrix B(n, edges);
                A = std::move(B);
                break;
        }
        case 's':
        {
                adjMatrix B(n, 's');
                A = std::move(B);
                break;
        }
        case 'b':
        {
                adjMatrix B(n, deg, 'b');
                A = std::move(B);
                break;

        }
        }

        krylov_dim = std::min(n-1, krylov_dim);

        edges = A.get_edges();

        std::cout << "n: " << A.get_n() << '\n';
        std::cout << "edges: " << A.get_edges() << '\n';
        std::cout << "krylov dimension: " << krylov_dim << '\n';

        /*std::cout << "A: \n"
                  << A << '\n';
                  */
        //A.print_full();
        std::vector<double> x(n, 1);

        //assert(krylov_dim <= n);

        lanczosDecomp L(A, krylov_dim, &x[0]);
        //std::cout << L;

        eigenDecomp E(L);
        //std::cout << E;
        multOut(L, E, A);

        std::cout << '\n';

        L.get_ans();
        std::cout << '\n';
        return 0;
}
