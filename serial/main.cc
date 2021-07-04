#include "adjMatrix.h"
#include "eigen.h"
#include "lanczos.h"
//#include <cassert>

template <typename T>
void print_vec(std::vector<T> v) {
        for (auto i = 0u; i < v.size(); ++i) {
                std::cout << v.at(i) << '\n';
        }
}

int main(int argc, char *argv[]){       
        if (argc != 2) {
                std::cerr << "Usage: ./adj file.txt\n";
                exit(EXIT_FAILURE);
        }
        std::string filename {argv[1]};
        
        std::ifstream fs;
        fs.open(filename);
        //assert (!fs.fail());
        
        unsigned n, edges;

        fs >> n >> n >> edges;
        
        adjMatrix A(n, edges, fs);
        
        std::cout << "n: " << A.get_n() << '\n';
        
        std::cout << "A: \n";
        A.print_full();
        
        std::vector<double> x(n, 1);
        
        unsigned krylov_dim {3};
        //assert(krylov_dim <= n);
        
        lanczosDecomp lanc(A, krylov_dim, x);

        std::cout << lanc;

        return 0;
}
        /*
        std::cout << "n:\t" << A.get_n() << std::endl;
        std::cout << "edges:\t" << A.get_edges() << std::endl;
        
        std::cout << '\n';
        
        std::cout << "A: \n" << A;

        std::cout << '\n';
        
        std::cout << "A: \n";
        A.print_full();

        std::vector<double> x(A.get_n(), 1.3);
        std::vector<double> v(A.get_n(), 0);
        
        std::cout << '\n';

        std::cout << "x: \n";
        print_vec(x);
        
        std::cout << '\n';
        
        sparse_adj_mat_vec_mult(A, x, v);

        std::cout << "Ax: \n";
        print_vec(v);
        
        std::cout << '\n';
        
        sparse_adj_mat_vec_mult(A, v, x);
        
        std::cout << "AAx: \n";
        print_vec(x);

        return 0;
}
*/
