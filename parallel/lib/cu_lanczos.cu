#include "cu_lanczos.h"

#define BLOCK_SIZE 32

void lanczosDecomp::cu_decompose()
{

    unsigned long n{A.get_n()};
    unsigned long *IA_d, *JA_d;
    double *Q_raw_d, *v_d, *alpha_d, *beta_d, *tmp_d;

    unsigned block_size{32}, num_blocks{static_cast<unsigned>(n) / block_size + 1};
    dim3 blocks{num_blocks}, threads{block_size};
    
    std::cout << num_blocks << " block(s) ";
    std::cout << "Running with "<< BLOCK_SIZE << " threads per block\n";

    double *x_normed{new double[n]};
    double x_norm = norm(x);

    /*
    std::cout << "x\n";
    for (auto k = 0u; k < n; k++)
        std::cout << x[k] << '\n';
*/

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

    for (auto k = 0u; k < krylov_dim; k++)
    {
        std::vector<double> tmp_vec(n);
        std::vector<long unsigned> tmp2(n);

        // v = A*Q(:,j)
        cu_spMV1<double, unsigned long><<<blocks, threads>>>(IA_d, JA_d, n, Q_d_ptr[i], v_d);
        
        /*
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
*/
        // alpha = v*Q(:,j)
        
        if (num_blocks==1) { 
            cu_dot_prod<double, BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(double)>>>(v_d, Q_d_ptr[i], n, &alpha_d[k]);
        /*double alpha_tmp{-11};
        cudaMemcpy(&alpha_tmp, &alpha_d[k], sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Got alpha=" << alpha_tmp << " from card\n";
        */
        } else {
            cu_dot_prod<double, BLOCK_SIZE><<<num_blocks/2, threads, BLOCK_SIZE*sizeof(double)>>>(v_d, Q_d_ptr[i], n, tmp_d);
            cu_reduce<double, BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(double)>>>(tmp_d, num_blocks, &alpha_d[k]);
        }



        // v = v - alpha*Q(:,j)
        cu_dpax<double><<<blocks, threads>>>(v_d, &alpha_d[k], Q_d_ptr[i], n);
        /*
        cudaMemcpy(&tmp_vec[0], &v_d[0], sizeof(double) * n, cudaMemcpyDeviceToHost);
        std::cout << "v -= alpha*Q_s: \n";
        std::for_each(tmp_vec.begin(), tmp_vec.end(), [](double a)
                      { std::cout << a << '\n'; });
*/
        if (k > 0)
        {
            // v = v - beta*Q(:,j-1)
            cu_dpax<double><<<blocks, threads>>>(v_d, &beta_d[k - 1], Q_d_ptr[1 - i], n);
        }

        if (k < krylov_dim - 1)
        {
            // beta[j] = norm(v)
            if (num_blocks==1) {
                cu_norm_sq_sqrt<double, BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(double)>>>(v_d, n, &beta_d[k]);
            } else {
                cu_norm_sq<double, BLOCK_SIZE><<<num_blocks/2, threads, BLOCK_SIZE*sizeof(double)>>>(v_d, n, tmp_d);
                cu_reduce_sqrt<double,BLOCK_SIZE><<<1, threads, BLOCK_SIZE*sizeof(double)>>>(tmp_d, num_blocks, &beta_d[k]);
            }
            cu_dvexda<double><<<blocks, threads>>>(Q_d_ptr[1 - i], &beta_d[k], v_d, n);
        }

        cudaMemcpy(&tmp[0], Q_d_ptr[i], sizeof(double) * n, cudaMemcpyDeviceToHost);

        for (auto j = 0u; j < n; j++)
            Q[k + j * krylov_dim] = tmp[j];
        /*
*/
        i = 1 - i;
    }
    cudaMemcpy(alpha, alpha_d, sizeof(double) * krylov_dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(beta, beta_d, sizeof(double) * (krylov_dim - 1), cudaMemcpyDeviceToHost);

/*
    std::cout << "cu_Q:\n";
    for (int i=0;i<krylov_dim;i++) {
        for (int j=0;j<krylov_dim;j++)
            std::cout << Q[i*krylov_dim+j] << " ";
        std::cout << '\n';
    }

    std::cout << "\ncu_Alpha:\n";
    for (int i=0;i<krylov_dim;i++) std::cout << alpha[i] << " ";
    
    std::cout << "\n\ncu_Beta:\n";
    for (int i=0;i<krylov_dim-1;i++) std::cout << beta[i] << " ";
    std::cout << "\n\n";
*/

    cudaFree(IA_d);
    cudaFree(JA_d);
    cudaFree(v_d);
    cudaFree(Q_raw_d);
    cudaFree(alpha_d);
    cudaFree(beta_d);
    cudaFree(tmp_d);
}


void lanczosDecomp::get_ans() const
{
        std::cout << "Answer vector:\n";

        for (auto i = 0u; i < A.n; i++)
                std::cout << std::setprecision(20) << ans[i] << '\n';
}

void lanczosDecomp::decompose()
{
        long unsigned n{A.get_n()}, i{0u};
        double *v{new double[n]};
        double *Q_raw(new double[2 * n]);
        double *Q_s[2]{Q_raw, &Q_raw[n]}; // Tmp contiguous columns to use before storing
                                          // in row-major matrix
/*
        std::cout << "A first edges: \n";
        for (auto j=0u;j<10;j++) 
        std::cout << A.row_idx[j] << " " << A.col_idx[j] << '\n';
        
        std::cout << "A last edges: \n";
        for (auto j=2*A.edge_count-10;j<2*A.edge_count;j++) 
        std::cout << A.row_idx[j] << " " << A.col_idx[j] << '\n';
  */      
        
        double x_norm = norm(x);

        for (auto k = 0u; k < n; k++)
                Q_s[i][k] = x[k] / x_norm;
        
        for (auto j = 0u; j < krylov_dim; j++)
        {

                // v = A*Q(:,j)
                spMV(A, Q_s[i], v);
                
                // alpha = v*Q(:,j)
                alpha[j] = inner_prod(v, Q_s[i], n);

                // v = v - alpha*Q(:,j)
                for (auto k = 0u; k < n; k++)
                        v[k] -= alpha[j] * Q_s[i][k];

                if (j > 0)
                {
                        // v = v - beta*Q(:,j-1)
                        for (auto k = 0u; k < n; k++)
                                v[k] -= beta[j - 1] * Q_s[1 - i][k];
                }

                if (j < krylov_dim - 1)
                {
                        beta[j] = norm(v);
                        for (auto k = 0u; k < n; k++)
                                Q_s[1 - i][k] = v[k] / beta[j];
                }

                // Copying the Q_s column into the Q matrix (implemented as a 1d row maj vector)
                for (auto k = 0u; k < n; k++)
                        Q[j + k * krylov_dim] = Q_s[i][k];

                i = 1 - i;
        }
        /*
        std::cout << "\nAlpha: ";
        for (auto j = 0u; j < krylov_dim; j++)
                std::cout << alpha[j] << " ";
        std::cout << "\nBeta: ";
        for (auto j = 0u; j < krylov_dim - 1; j++)
                std::cout << beta[j] << " ";
        std::cout << '\n';
        std::cout << '\n';
*/
/* PRINT OUT Q 
        std::cout << "\nQ\n";
        for (auto j = 0u; j < krylov_dim; j++)
        {
                for (auto k = 0u; k < krylov_dim; k++)
                        std::cout << Q[k + j * krylov_dim] << " ";
                std::cout << '\n';
        }*/
        delete[] v;
        delete[] Q_raw;
}

std::ostream &operator<<(std::ostream &os, const lanczosDecomp &D)
{
        auto n {D.A.get_n()};
        os << "\nAlpha: \n";
        for (auto i = 0u; i < n; i++)
                os << D.alpha[i] << " ";

        os << "\n\nBeta: \n";
        for (auto i = 0u; i < n - 1; i++)
                os << D.beta[i] << " ";

        os << "\n\nQ:\n";
        for (auto i = 0u; i < n - 1; i++)
        {
                os << D.Q[i] << " ";
                if (i % D.krylov_dim == D.krylov_dim - 1)
                        os << '\n';
        }

        os << '\n';
        return os;
}

void lanczosDecomp::check_ans(const double *analytic_ans) const
{
        std::vector<double> diff(A.n);
        for (auto i = 0u; i < A.n; i++)
        {
                diff[i] = std::abs(ans[i] - analytic_ans[i]);
        }
        auto max_it = std::max_element(diff.begin(), diff.end());
        auto max_idx = std::distance(diff.begin(), max_it);
        std::cout << "\nMax difference of " << *max_it
                  << " found at index\n\tlanczos[" << max_idx << "] \t\t\t= " << ans[max_idx]
                  << "\n\tanalytic_ans[" << max_idx << "] \t\t= " << analytic_ans[max_idx] << '\n';

        std::cout << "\nTotal norm of differences\t= " << std::setprecision(20) << (&diff[0]) << '\n';
        std::cout << "Relative norm of differences\t= " << std::setprecision(20)<< norm(&diff[0])/norm(analytic_ans) << '\n';
}

void lanczosDecomp::check_ans(lanczosDecomp & L) const
{     
        /*
        unsigned width {15};
        std::cout <<std::setw(width) << "Serial" << std::setw(width) << "CUDA" << std::endl;
        for (int i=0;i<5;i++) {
          std::cout << std::setw(width) << ans[i] << std::setw(width) << L.ans[i] << std::endl;
        }
        */
        std::vector<double> diff(A.n);
        for (auto i = 0u; i < A.n; i++)
        {
                diff[i] = std::abs(ans[i] - L.ans[i]);
        }
        auto max_it = std::max_element(diff.begin(), diff.end());
        auto max_idx = std::distance(diff.begin(), max_it);
        std::cout << "\nMax difference of " << *max_it
                  << " found at index\n"<<std::setw(15)<<"serial_ans[" << max_idx << "] = " <<std::setprecision(10)<<std::setw(15)<< ans[max_idx] <<"\n"
                  <<std::setw(15)<<"cuda_ans[" << max_idx << "] = " <<std::setprecision(10)<<std::setw(15) << L.ans[max_idx] << '\n' << std::endl;

        std::cout << std::setw(30) << std::left << "Total norm of differences" << "=" << std::right << std::setprecision(20) <<  std::setw(30) << norm(&diff[0]) << std::endl;
        std::cout << std::setw(30) << std::left << "Relative norm of differences"<< "=" << std::right << std::setprecision(20) << std::setw(30) << norm(&diff[0])/norm(L.ans) << std::endl;
}
/*
// Doesn't work! (doesn't give better accuracy)
void lanczosDecomp::reorthog() {
        double * tau {new double [krylov_dim]};
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, A.n, krylov_dim, Q, krylov_dim, tau);
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, A.n, krylov_dim, krylov_dim, Q, krylov_dim, tau);
        delete[] tau;
}
*/
double lanczosDecomp::inner_prod(const double *const v, const double *const w, const long unsigned N) const
{
        double ans{0.0};
        for (auto i = 0u; i < N; i++)
        {
                ans += v[i] * w[i];
        }
        return ans;
}
