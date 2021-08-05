#ifndef HELPERS_H_32423423
#define HELPERS_H_32423423

#include <string>
#include <unistd.h>

int parseArguments(int, char **, std::string &, long unsigned &, bool &, long unsigned &, long unsigned &, long unsigned &);
template <typename T>
void diff_arrays(const T * const, const T * const, const unsigned n, T & , unsigned & );
template <typename T>
T norm(const T * const a, const unsigned n);
#endif
