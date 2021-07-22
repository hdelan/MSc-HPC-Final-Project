#include "adjMatrix.h"

void adjMatrix::populate_sparse_matrix(std::ifstream &f)
{
        // This variable holds the number of nonzero values
        // above the diagonal in each column. This is needed in
        // order to
        unsigned row, col;

        std::vector<std::vector<unsigned>> cols(n);
        std::vector<std::vector<unsigned>> rows(n);

        // Putting the upper triangular parts into rows, cols
        for (auto i = 0u; i < edge_count; ++i)
        {
                f >> col >> row;
                col--;
                row--;
                rows[row].emplace_back(col);
                cols[col].emplace_back(row);
        }

        for (auto i = 0u; i < n; i++)
                rows[i].insert(rows[i].begin(), cols[i].begin(), cols[i].end());

        unsigned offset{0u};
        for (auto i = 0u; i < n; ++i)
        {
                if (rows[i].size() != 0)
                {
                        for (auto j = 0u; j < rows[i].size(); j++) row_idx[offset+j] = rows[i][j];
                        std::vector<unsigned> inserter(rows[i].size(), i);
                        for (auto j = 0u; j < inserter.size(); j++) col_idx[offset+j] = inserter[j];
                        offset += rows[i].size();
                }
        }
}

std::ostream &operator<<(std::ostream &os, const adjMatrix &A)
{
        for (auto i = 0u; i < A.edge_count * 2; ++i)
        {
                os << "(" << A.row_idx[i] << ", " << A.col_idx[i] << ")\n";
        }
        return os;
}

void adjMatrix::print_full()const{
        auto i {0u}, j {0u};
        while (i < n*n) {
                if (row_idx[j] == i%n && col_idx[j] == i/n) {
                        std::cout << " 1";
                        j++;
                } else {
                        std::cout << " *";
                }
                if (i % n == n-1) std::cout << '\n';
                i++;
        }
}
