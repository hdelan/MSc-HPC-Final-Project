#include "lanczos.h"

void lanczosDecomp::cu_decompose()
{

    unsigned n{static_cast<unsigned>(A.get_n())};
    unsigned long *IA_d, *JA_d;
    double *Q_raw_d, *v_d, *alpha_d, *beta_d, *tmp_d;

    unsigned block_size{32}, num_blocks{n / block_size + 1};
    dim3 blocks{num_blocks}, threads{block_size};

    double *x_normed{new double[n]};
    double x_norm = norm(x);

    for (auto k = 0u; k < n; k++)
        x_normed[k] = x[k] / x_norm;

    cudaMalloc((void **)&IA_d, sizeof(long unsigned) * (n + 1));
    cudaMalloc((void **)&JA_d, sizeof(long unsigned) * 2 * A.edge_count);
    cudaMalloc((void **)&v_d, sizeof(double) * n);
    cudaMalloc((void **)&Q_raw_d, sizeof(double) * n * 2);
    cudaMalloc((void **)&alpha_d, sizeof(double) * krylov_dim);
    cudaMalloc((void **)&beta_d, sizeof(double) * (krylov_dim - 1));
    cudaMalloc((void **)&tmp_d, sizeof(double) * (num_blocks));

    double *Q_s[2] = {&Q_raw_d[0], &Q_raw_d[n]};

    cudaMemcpy(x_normed, Q_s[0], sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(A.row_offset, IA_d, sizeof(long unsigned) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(A.col_idx, JA_d, sizeof(long unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);

    std::vector<double> tmp(n);

    int i{0};

    std::cout << "1\n";

    for (auto k = 0u; k < krylov_dim; k++)
    {

        // v = A*Q(:,j)
        cu_spMV1<<<blocks, threads>>>(IA_d, JA_d, n, Q_s[i], v_d);

        // alpha = v*Q(:,j)
        cu_dot_prod<<<blocks, threads>>>(v_d, Q_s[i], n, tmp_d);
        cu_reduce<<<1, threads>>>(tmp_d, num_blocks, &alpha_d[k]);

        // v = v - alpha*Q(:,j)
        cu_dpax<<<blocks, threads>>>(v_d, alpha_d[k], Q_s[i], n);

        if (k > 0)
        {
            // v = v - beta*Q(:,j-1)
            cu_dpax<<<blocks, threads>>>(v_d, beta_d[k - 1], Q_s[1 - i], n);
        }

        if (k < krylov_dim - 1)
        {
            // beta[j] = norm(v)
            cu_norm_sq<<<blocks, threads>>>(v_d, n, tmp_d);
            cu_reduce_sqrt<<<1, threads>>>(tmp_d, num_blocks, &beta_d[k]);

            cu_dvexda<<<blocks, threads>>>(Q_s[1 - i], beta_d[k], v_d, n);
        }

        cudaMemcpy(Q_s[i], &tmp[0], sizeof(double) * n, cudaMemcpyDeviceToHost);

        for (auto j = 0u; j < n; j++)
            Q[k + j * krylov_dim] = tmp[j];

        i = 1 - i;
    }
    cudaMemcpy(alpha_d, alpha, sizeof(double) * krylov_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(beta_d, beta, sizeof(double) * (krylov_dim-1), cudaMemcpyDeviceToHost);

    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(v_d);
    cudaFree(Q_raw_d);
    cudaFree(alpha_d);
    cudaFree(beta_d);
    cudaFree(tmp_d);
}