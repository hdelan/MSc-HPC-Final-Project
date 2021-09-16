/**
 * \file:        adjMatrix.cc
 * \brief:       An adjacency matrix helper constructor
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-15
 */

#include "adjMatrix.h"
#include "edge.h"

void adjMatrix::populate_sparse_matrix(std::ifstream &f)
{
  std::set<Edge> edges;
  // This variable holds the number of nonzero values
  // above the diagonal in each column. This is needed in
  // order to
  unsigned row, col;
  std::string line;

  // Putting both upper, lower triangular parts into rows, cols
  //while (std::getline(f, line)) {
  for (auto i=0u;i<edge_count;i++) {
    f >> col >> row;
    //std::cout << col << " " << row << '\n';
    edges.emplace(Edge(--row, --col));
    edges.emplace(Edge(col, row));
  }

  auto i{0u};
  auto prev_row {0u};
  for (auto it = edges.begin(); it != edges.end(); it++)
  {
    //std::cout << "(" << it->n1 << "," << it->n2 << ")\n";
    while (prev_row != it->n1) row_offset[++prev_row] = i;
    col_idx[i++] = it->n2;
  }
  row_offset[n] = edges.size();
  edge_count = edges.size()/2;

}

void adjMatrix::write_matrix_to_file() {
  std::string dir {"../data/generated/"};
  std::string type {matrix_type};
  std::string filename {type+"n"+std::to_string(n)+"e"+std::to_string(edge_count)};
  filename = dir + filename;
  std::cout << "Filename: " << filename << '\n';
  std::ofstream f;
  f.open(filename);
  assert(!f.fail());
  f << n << " " << n << " " << edge_count << '\n';
  for (auto i=0u;i<n;i++) {
    for (auto j=row_offset[i];j<row_offset[i+1];j++) {
      if (i < col_idx[j]) f << col_idx[j]+1 << " " << i+1 << '\n';
    }
  }
  f.close();
}


void adjMatrix::generate_sparse_matrix(const char c)
{
  switch (c)
  {
    case 's':
      {
        row_offset = new unsigned[n*n + 1];
        col_idx = new unsigned[2 * (2 * n * n - n - 1)];
        edge_count = (2 * n * n - n - 1);
        n = n * n;
        //stencil_adj();
        break;
      }
    case 'b':
      {
        barabasi(barabasi_degree);
        break;
      }
    case 'r':
      {
        random_adj();
        break;
      }
  }
}


std::ostream &operator<<(std::ostream &os, const adjMatrix &A)
{
  os << "JA\n";
  for (auto i = 0u; i < A.edge_count * 2; ++i)
  {
    os << A.col_idx[i] << " ";
  }
  os << "\nIA\n";
  for (auto i = 0u; i < A.n+1; ++i)
  {
    os << A.row_offset[i] << " ";
  }

  return os;
}
/*
   void adjMatrix::print_full() const
   {
   auto i{0u}, j{0u};
   while (i < n * n)
   {
   if (row_idx[j] == i / n && col_idx[j] == i % n)
   {
   std::cout << " 1";
   j++;
   }
   else
   {
   std::cout << " *";
   }
   if (i % n == n - 1)
   std::cout << '\n';
   i++;
   }
   }


   void get_raw_upper_matrix(double * mat, adjMatrix & A) {

   std::vector<Edge> edges(A.edge_count);

// Get the upper triangular edges
for (auto i=0u;i<A.edge_count*2;i++){
if (A.row_idx[i] < A.col_idx[i]) 
edges.emplace_back(Edge(A.row_idx[i], A.col_idx[i]));

}

auto k {0u}, m {0u};
for (unsigned i=0u;i<A.n;i++) {
for (auto j=i;j<A.n;j++) {
if (Edge(i,j) == edges[k]) {
mat[m] = 1.0;
k++;
}
else {
mat[m] = 0.0;
} 
m++;
}
}
}

void get_raw_full_matrix(double * mat, adjMatrix & A) {

std::vector<Edge> edges(A.edge_count);

// Get the upper triangular edges
for (auto i=0u;i<A.edge_count*2;i++){
edges.emplace_back(Edge(A.row_idx[i], A.col_idx[i]));

}

auto k {0u}, m {0u};
for (unsigned i=0u;i<A.n;i++) {
for (auto j=0u;j<A.n;j++) {
if (Edge(i,j) == edges[k]) {
mat[m] = 1.0;
k++;
}
else {
mat[m] = 0.0;
} 
m++;
}
}
}

*/
