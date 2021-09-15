#ifndef WRITE_MATRIX_H1234
#define WRITE_MATRIX_H1234

#include <string>
#include <fstream>

#include "cu_lanczos.h"

template <typename T>
void write_ans(std::string filename, lanczosDecomp<T> & L) {
  std::ofstream fs;
  fs.open(filename);
  assert(!fs.fail());
  for (auto i=0u;i<L.A.get_n();i++)
    fs << L.ans[i] << '\n';
}

#endif
