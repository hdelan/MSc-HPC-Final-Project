 /**
 * \file:        check_ans.cu
 * \brief:       A function to compare the answer vectors of two lanczos decomposition objects
 * \author:      Hugh Delaney
 * \version:     
 * \date:        2021-09-16
 */
#include "check_ans.h"

template <typename T, typename U>
void check_ans(lanczosDecomp<T> &L1, lanczosDecomp<U> &L2)

{     
  auto n {L1.A.get_n()};
    std::vector<decltype(T()+U())> diff(n);
  for (auto i = 0u; i < n; i++)
  {
    diff[i] = std::abs(L1.ans[i] - L2.ans[i]);
  }
  auto max_it = std::max_element(diff.begin(), diff.end());
  auto max_idx = std::distance(diff.begin(), max_it);
  std::cout << "\nMax difference of " << *max_it << " (Relative difference: " << *max_it/L2.ans[max_idx] << ") "
    << "found at index:\n"<<std::setw(15)<<"serial_ans[" << max_idx << "] = " <<std::setprecision(10)<<std::setw(15)<< L1.ans[max_idx] <<"\n"
    <<std::setw(15)<<"cuda_ans[" << max_idx << "] = " <<std::setprecision(10)<<std::setw(15) << L2.ans[max_idx] << '\n' << std::endl;

  std::cout << std::setw(30) << std::left << "Total norm of differences" << "=" << std::right << std::setprecision(20) <<  std::setw(30) << norm(&diff[0],n) << std::endl;
  std::cout << std::setw(30) << std::left << "Relative norm of differences"<< "=" << std::right << std::setprecision(20) << std::setw(30) << norm(&diff[0],n)/norm(L2.ans,n) << std::endl;
}

template void check_ans(lanczosDecomp<float> &, lanczosDecomp<float>&);
template void check_ans(lanczosDecomp<double> &, lanczosDecomp<float>&);
template void check_ans(lanczosDecomp<float> &, lanczosDecomp<double>&);
template void check_ans(lanczosDecomp<double> &, lanczosDecomp<double>&);
