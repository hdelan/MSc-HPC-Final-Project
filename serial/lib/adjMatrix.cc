#include "adjMatrix.h"


void adjMatrix::print_full() {
        unsigned v_idx;
        for (auto i = 0u; i < n; ++i) {
                v_idx = 0;
                for (auto j = 0u; j < n; ++j) {
                        if (v_idx >= rows.at(i).size() || j < rows.at(i).at(v_idx)) {
                                std::cout << "* ";
                        } else {
                                std::cout << "1 ";
                                v_idx++;
                        }
                }
                std::cout << '\n';
        }
}

void adjMatrix::populate_sparse_matrix(std::ifstream & f) {
        // This variable holds the number of nonzero values
        // above the diagonal in each column. This is needed in 
        // order to 
        unsigned row, col;

        std::vector<std::vector<unsigned>> cols(n);

        // Putting the upper triangular parts into rows, cols
        for (auto i = 0u; i < edge_count; ++i) {
                f >> col >> row;
                col--; row--;
                rows.at(row).push_back(col);
                cols.at(col).push_back(row);
        }

        for (auto i=0u;i<n;i++)
                rows.at(i).insert(rows.at(i).begin(), cols.at(i).begin(), cols.at(i).end());

        for (auto i = 0u; i < n; ++i) {
                unsigned offset {0u};
                if (rows.at(i).size() != 0) {
                        idx[0].insert(idx[0].begin()+offset, rows.at(i).begin(), rows.at(i).end());
                        std::vector<unsigned> insert(rows.at(i).size(), i);
                        idx[1].insert(idx[1].begin()+offset, insert.begin(), insert.end());
                        offset += rows.at(i).size();
                }
        }
}

