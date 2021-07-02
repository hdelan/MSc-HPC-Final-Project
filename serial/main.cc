#include "matrix_exp.h"

int main(int argc, char *argv[])
{       
        try {
                adjMatrix A(argv[1]);
        } catch (std::exception e) {
                std::cerr << "Reading file failed. Caught: " << e.what();
        } catch (std::bad_alloc b) {
                std::cerr << "Allocating space failed. Caught: " << b.what();
        }

        std::cout << "n: " << A.get_n() << std::endl;
        std::cout << "edges: " << A.get_edges() << std::endl;
        std::cout << "A: \n" << A << std::endl;
        return 0;
}

