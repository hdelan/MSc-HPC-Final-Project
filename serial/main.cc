#include "adjMatrix.h"
#include "eigen.h"
#include "lanczos.h"
#include "helpers.h"
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
        
        //std::cout << "A: \n";
        //A.print_full();
        
        std::vector<double> x(n, 1);
        
        //assert(krylov_dim <= n);
        
        lanczosDecomp L(A, krylov_dim, x);
        //std::cout << L;
        
        eigenDecomp E(L);
        //std::cout << E;

        multOut(L, E, A);
        
        std::cout << '\n';

        A.get_ans();

        std::cout << '\n';
        
        return 0;
}

