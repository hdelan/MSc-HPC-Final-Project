#include "cu_lanczos.h"

#define BLOCK_SIZE 32

void lanczosDecomp::cu_decompose()
{

    unsigned long n{A.get_n()};
    unsigned long *IA_d, *JA_d;
    double *Q_raw_d, *v_d, *alpha_d, *beta_d, *tmp_d;

    unsigned block_size{32}, num_blocks{static_cast<unsigned>(n) / block_size + 1};
    dim3 blocks{num_blocks}, threads{block_size};
    
    std::cout << BLOCK_SIZE << " threads per block\n";
    std::cout << num_blocks << " block(s)\n";

    double *x_normed{new double[n]};
    double x_norm = norm(x);
    std::cout << "x\n";
    for (auto k = 0u; k < n; k++)
        std::cout << x[k] << '\n';


    for (auto k = 0u; k < n; k++)
        x_normed[k] = x[k] / x_norm;

    cudaMalloc((void **)&IA_d, sizeof(long unsigned) * (n + 1));
    cudaMalloc((void **)&JA_d, sizeof(long unsigned) * 2 * A.edge_count);
    cudaMalloc((void **)&v_d, sizeof(double) * n);
    cudaMalloc((void **)&Q_raw_d, sizeof(double) * n * 2);
    cudaMalloc((void **)&alpha_d, sizeof(double) * krylov_dim);
    cudaMalloc((void **)&beta_d, sizeof(double) * (krylov_dim - 1));
    cudaMalloc((void **)&tmp_d, sizeof(double) * (num_blocks));

    double *Q_d_ptr[2] = {&Q_raw_d[0], &Q_raw_d[n]};

    cudaMemcpy(Q_d_ptr[0], x_normed, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(IA_d, A.row_offset, sizeof(long unsigned) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(JA_d, A.col_idx, sizeof(long unsigned) * 2 * A.edge_count, cudaMemcpyHostToDevice);

    std::vector<double> tmp(n);

    int i{0};

    std::cout << "1\n";
    for (auto k = 0u; k < krylov_dim; k++)
    {
        std::vector<double> tmp_vec(n);
        std::vector<long unsigned> tmp2(n);

        // v = A*Q(:,j)
        cu_spMV1<double, unsigned long><<<blocks, threads>>>(IA_d, JA_d, n, Q_d_ptr[i], v_d);

        cudaMemcpy(&tmp2[0], &IA_d[0], sizeof(double) * n, cudaMemcpyDeviceToHost);
        std::cout << "IA: \n";
        std::for_each(tmp2.begin(), tmp2.end(), [](long unsigned a)
                      { std::cout << a << '\n'; });
        cudaMemcpy(&tmp_vec[0], &v_d[0], sizeof(double) * n, cudaMemcpyDeviceToHost);
        std::cout << "cu_SPMV product: \n";
        std::for_each(tmp_vec.begin(), tmp_vec.end(), [](double a)
                      { std::cout << a << '\n'; });
        cudaMemcpy(&tmp_vec[0], Q_d_ptr[0], sizeof(double) * n, cudaMemcpyDeviceToHost);
        std::cout << "Q_s: \n";
        std::for_each(tmp_vec.begin(), tmp_vec.end(), [](double a)
                      { std::cout << a << '\n'; });

        // alpha = v*Q(:,j)
        
        if (num_blocks==1) { 
            printf("In here!\n");
            cu_dot_prod<double, BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(double)>>>(v_d, Q_d_ptr[i], n, &alpha_d[k]);
        double alpha_tmp{-11};
        cudaMemcpy(&alpha_tmp, tmp_d, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Got alpha=" << alpha_tmp << " from card\n";
        } else {
            cu_dot_prod<double, BLOCK_SIZE><<<std::max(1u,num_blocks/2), threads, BLOCK_SIZE*sizeof(double)>>>(v_d, Q_d_ptr[i], n, tmp_d);
            cu_reduce<double, BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(double)>>>(tmp_d, num_blocks, &alpha_d[k]);
        }



        /*
        // v = v - alpha*Q(:,j)
        cu_dpax<double><<<blocks, threads>>>(v_d, alpha_d[k], Q_d_ptr[i], n);

        if (k > 0)
        {
            // v = v - beta*Q(:,j-1)
            cu_dpax<double><<<blocks, threads>>>(v_d, beta_d[k - 1], Q_s[1 - i], n);
        }

        if (k < krylov_dim - 1)
        {
            // beta[j] = norm(v)
            cu_norm_sq<double, BLOCK_SIZE><<<blocks, threads>>>(v_d, n, tmp_d);
            cu_reduce_sqrt<double,BLOCK_SIZE><<<1, threads>>>(tmp_d, num_blocks, &beta_d[k]);

            cu_dvexda<double><<<blocks, threads>>>(Q_s[1 - i], beta_d[k], v_d, n);
        }

        cudaMemcpy(Q_s[i], &tmp[0], sizeof(double) * n, cudaMemcpyDeviceToHost);

        for (auto j = 0u; j < n; j++)
            Q[k + j * krylov_dim] = tmp[j];
*/
        i = 1 - i;
    }
    cudaMemcpy(alpha_d, alpha, sizeof(double) * krylov_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(beta_d, beta, sizeof(double) * (krylov_dim - 1), cudaMemcpyDeviceToHost);

    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(v_d);
    cudaFree(Q_raw_d);
    cudaFree(alpha_d);
    cudaFree(beta_d);
    cudaFree(tmp_d);
}