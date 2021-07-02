#ifndef MATRIX_EXP_H_9324
#define MATRIX_EXP_H_9324

#include <iostream>
#include <iomanip>
#include <fstream>

class adjMatrix {
        public:
                adjMatrix() = delete;
                adjMatrix(std::string filename) {
                        std::ifstream f;
                        f.open(filename);
                        if (f.fail()) throw std::exception();
                        
                        f >> n >> n >> edge_count;
                        row = new unsigned[edge_count];
                        col = new unsigned[edge_count];
                        for (auto i = 0u; i < edge_count; ++i) {
                                f >> col[i] >> row[i];
                                col[i] -= 1;
                                row[i] -= 1;
                        }
                };
                adjMatrix(unsigned N, unsigned edges) : n {N}, edge_count {edges} {
                        std::random_device rd;
                };
                adjMatrix(adjMatrix &) = delete;
                adjMatrix & operator=(adjMatrix &) = delete;
                ~adjMatrix() {
                        delete[] row;
                        delete[] col;
                }

                unsigned get_n() { return n;};
                unsigned get_edges() { return edge_count;};
                friend std::ostream& operator<<(std::ostream & os, const adjMatrix & A) {
                        for (auto i = 0u; i < A.edge_count; ++i) {
                                os << "(" << A.row[i] << ", " << A.col[i] << ")\n";
                        }
                        return os;
                };

        private:
                unsigned * row;
                unsigned * col;
                unsigned edge_count;
                unsigned n;
};

#endif
