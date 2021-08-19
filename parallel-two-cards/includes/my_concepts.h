#ifndef MY_CONCEPTS203499
#define MY_CONCEPTS203499

#include <type_traits>

template <typename T>
concept Integral=std::is_integral<T>::value;

#endif
