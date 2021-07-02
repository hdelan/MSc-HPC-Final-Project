#include "matrix_exp.h"
#include <cassert>


int main(int argc, char *argv[])
{       
        if (argc != 2) {
                std::cerr << "Usage: ./adj file.txt\n";
                exit(EXIT_FAILURE);
        }
        std::string filename {argv[1]};

        std::ifstream fs;
        assert (!fs.fail());
        adjMatrix A(fs);

        std::cout << "n: " << A.get_n() << std::endl;
        std::cout << "edges: " << A.get_edges() << std::endl;
        std::cout << "A: \n" << A << std::endl;
        return 0;
}

