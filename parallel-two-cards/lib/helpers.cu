#include "helpers.h"

#include <numeric>
#include <vector>
#include <cmath>

void cuda_start_timer(cudaEvent_t &start, cudaEvent_t &end)
{
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
}

float cuda_end_timer(cudaEvent_t &start, cudaEvent_t &end)
{
    cudaEventRecord(end, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
    float time_taken;
    cudaEventElapsedTime(&time_taken, start, end);
    return time_taken * 0.001;
}

int parseArguments(int argc, char *argv[], std::string &filename, long unsigned &krylov_dim, bool &verbose, long unsigned &n, long unsigned &bar_deg, long unsigned &E)
{
        int c;

        while ((c = getopt(argc, argv, "k:f:b:n:e:v")) != -1)
        {
                switch (c)
                {
                case 'f':
                        filename = optarg;
                        break;
                case 'k':
                        krylov_dim = atoi(optarg);
                        break;
                case 'b':
                        bar_deg = atoi(optarg);
                        break;
                case 'n':
                        n = atoi(optarg);
                        break;
                case 'e':
                        E = atoi(optarg);
                        break;
                case 'v':
                        verbose = true;
                        break;
                default:
                        fprintf(stderr, "Invalid option given\n");
                        return -1;
                }
        }
        return 0;
}

template <typename T>
void diff_arrays(const T *const a, const T *const b, const unsigned n, T &relative_error, unsigned &max_entry)
{
        std::vector<T> diff(n);
        max_entry = 0u;
        for (auto i = 0u; i < n; i++)
        {
                diff[i] = a[i] - b[i];
                if (diff[i] > diff[max_entry])
                        max_entry = i;
        }
        relative_error = std::sqrt(std::inner_product(diff.begin(), diff.end(),diff.begin(),0))/norm(a, n);
}

template <typename T>
T norm(const T *const a, const unsigned n)
{
        T ans{0};
        for (auto i = 0u; i < n; i++)
        {
                ans += a[i] * a[i];
        }
        return std::sqrt(ans);
}

template float norm<float>(const float *const a, const unsigned n);
template double norm<double>(const double *const a, const unsigned n);

template void diff_arrays<float>(const float *const, const float *const, const unsigned n, float &, unsigned &);
template void diff_arrays<double>(const double *const, const double *const, const unsigned n, double &, unsigned &);
