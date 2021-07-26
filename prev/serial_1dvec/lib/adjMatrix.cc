#include "adjMatrix.h"

void adjMatrix::populate_sparse_matrix(std::ifstream & f) {
        // This variable holds the number of nonzero values
        // above the diagonal in each column. This is needed in 
        // order to 
        unsigned row, col;

        std::vector<std::vector<unsigned>> cols(n);
        std::vector<std::vector<unsigned>> rows(n);

        // Putting the upper triangular parts into rows, cols
        for (auto i = 0u; i < edge_count; ++i) {
                f >> col >> row;
                col--; row--;
                rows[row].emplace_back(col);
                cols[col].emplace_back(row);
        }

        for (auto i=0u;i<n;i++)
                rows[i].insert(rows[i].begin(), cols[i].begin(), cols[i].end());

        for (auto i = 0u; i < n; ++i) {
                unsigned offset {0u};
                if (rows[i].size() != 0) {
                        row_idx.insert(row_idx.begin()+offset, rows[i].begin(), rows[i].end());
                        std::vector<unsigned> insert(rows[i].size(), i);
                        col_idx.insert(col_idx.begin()+offset, insert.begin(), insert.end());
                        offset += rows[i].size();
                }
        }
}

std::ostream& operator<<(std::ostream & os, const adjMatrix & A) {
        for (auto i = 0u; i < A.edge_count*2; ++i) {
                os << "(" << A.row_idx[i] << ", " << A.col_idx[i] << ")\n";
        }
        return os;
}
