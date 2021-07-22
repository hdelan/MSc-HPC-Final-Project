#include "lib/adjMatrix.h"
#include "lib/eigen.h"
#include "lib/lanczos.h"
#include "lib/helpers.h"
#include "lib/sparse_mult.h"
#include "lib/multiplyOut.h"
#include <cassert>

int main(int argc, char *argv[]){       
        
        std::string filename {"../data/file.txt"};
        unsigned krylov_dim {3};
        
        parseArguments(argc, argv, filename, krylov_dim);

        std::ifstream fs;
        fs.open(filename);
        //assert (!fs.fail());
        
        unsigned n, edges;

        fs >> n >> n >> edges;
        
        adjMatrix A(n, edges, fs);
        std::cout << "n: " << A.get_n() << '\n';
        std::cout << "krylov dimension: " << krylov_dim << '\n';
        
        //std::cout << "A: \n" << A;
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

