// gpu_solver.cu -- CUDA BiCGSTAB with cuSPARSE/cuBLAS
#ifdef USE_CUDA

#include "twofluid/gpu_solver.hpp"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>

namespace twofluid {

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " : " << cudaGetErrorString(err) << "\n"; \
        GPUSolveResult fail_result; \
        fail_result.converged = false; \
        return fail_result; \
    } \
} while(0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t st = call; \
    if (st != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ \
                  << " : status=" << static_cast<int>(st) << "\n"; \
        GPUSolveResult fail_result; \
        fail_result.converged = false; \
        return fail_result; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = call; \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                  << " : status=" << static_cast<int>(st) << "\n"; \
        GPUSolveResult fail_result; \
        fail_result.converged = false; \
        return fail_result; \
    } \
} while(0)

GPUSolveResult gpu_bicgstab(
    const int n,
    const int* row_ptr, const int* col_idx, const double* vals,
    const double* rhs, const double* x0,
    int nnz, double tol, int maxiter) {

    GPUSolveResult result;
    result.converged = false;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Device memory pointers
    double *d_vals = nullptr, *d_x = nullptr, *d_b = nullptr;
    double *d_r = nullptr, *d_r_hat = nullptr, *d_p = nullptr;
    double *d_v = nullptr, *d_s = nullptr, *d_t = nullptr;
    int *d_row_ptr = nullptr, *d_col_idx = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_hat, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_s, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_t, n * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_vals, vals, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, rhs, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x0, n * sizeof(double), cudaMemcpyHostToDevice));

    // Create cuSPARSE and cuBLAS handles
    cusparseHandle_t cusparse = nullptr;
    cublasHandle_t cublas = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&cusparse));
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Create sparse matrix descriptor (CSR)
    cusparseSpMatDescr_t matA = nullptr;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, n, n, nnz,
                                      d_row_ptr, d_col_idx, d_vals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // Allocate SpMV buffer
    double alpha_sp = 1.0, beta_sp = 0.0;
    cusparseDnVecDescr_t vecTmpIn = nullptr, vecTmpOut = nullptr;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecTmpIn, n, d_p, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecTmpOut, n, d_v, CUDA_R_64F));

    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_sp, matA, vecTmpIn, &beta_sp, vecTmpOut,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    void* buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer, bufferSize));

    cusparseDestroyDnVec(vecTmpIn);
    cusparseDestroyDnVec(vecTmpOut);

    // Lambda for SpMV: out = A * in
    auto spmv = [&](double* d_in, double* d_out) {
        cusparseDnVecDescr_t vIn = nullptr, vOut = nullptr;
        cusparseCreateDnVec(&vIn, n, d_in, CUDA_R_64F);
        cusparseCreateDnVec(&vOut, n, d_out, CUDA_R_64F);
        double a = 1.0, b = 0.0;
        cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &a, matA, vIn, &b, vOut,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
        cusparseDestroyDnVec(vIn);
        cusparseDestroyDnVec(vOut);
    };

    // Compute r = b - A*x0
    spmv(d_x, d_r);  // r = A*x0
    double neg_one = -1.0, one = 1.0;
    CUBLAS_CHECK(cublasDscal(cublas, n, &neg_one, d_r, 1));   // r = -A*x0
    CUBLAS_CHECK(cublasDaxpy(cublas, n, &one, d_b, 1, d_r, 1)); // r = b - A*x0

    // r_hat = r
    CUDA_CHECK(cudaMemcpy(d_r_hat, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));

    // Initialize p = 0, v = 0
    CUDA_CHECK(cudaMemset(d_p, 0, n * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_v, 0, n * sizeof(double)));

    double rho_old = 1.0, alpha_bc = 1.0, omega = 1.0;
    double b_norm;
    CUBLAS_CHECK(cublasDnrm2(cublas, n, d_b, 1, &b_norm));
    if (b_norm < 1e-30) b_norm = 1.0;

    int iter;
    for (iter = 0; iter < maxiter; ++iter) {
        double rho_new;
        CUBLAS_CHECK(cublasDdot(cublas, n, d_r_hat, 1, d_r, 1, &rho_new));

        if (std::abs(rho_new) < 1e-300) break;

        double beta = (rho_new / rho_old) * (alpha_bc / omega);

        // p = r + beta*(p - omega*v)
        double neg_omega = -omega;
        CUBLAS_CHECK(cublasDaxpy(cublas, n, &neg_omega, d_v, 1, d_p, 1)); // p -= omega*v
        CUBLAS_CHECK(cublasDscal(cublas, n, &beta, d_p, 1));               // p *= beta
        CUBLAS_CHECK(cublasDaxpy(cublas, n, &one, d_r, 1, d_p, 1));       // p += r

        // v = A*p
        spmv(d_p, d_v);

        double denom;
        CUBLAS_CHECK(cublasDdot(cublas, n, d_r_hat, 1, d_v, 1, &denom));
        if (std::abs(denom) < 1e-300) break;
        alpha_bc = rho_new / denom;

        // s = r - alpha*v
        CUDA_CHECK(cudaMemcpy(d_s, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));
        double neg_alpha = -alpha_bc;
        CUBLAS_CHECK(cublasDaxpy(cublas, n, &neg_alpha, d_v, 1, d_s, 1));

        double s_norm;
        CUBLAS_CHECK(cublasDnrm2(cublas, n, d_s, 1, &s_norm));
        if (s_norm / b_norm < tol) {
            CUBLAS_CHECK(cublasDaxpy(cublas, n, &alpha_bc, d_p, 1, d_x, 1));
            result.iterations = iter + 1;
            result.residual = s_norm / b_norm;
            result.converged = true;
            break;
        }

        // t = A*s
        spmv(d_s, d_t);

        double t_dot_t, t_dot_s;
        CUBLAS_CHECK(cublasDdot(cublas, n, d_t, 1, d_t, 1, &t_dot_t));
        CUBLAS_CHECK(cublasDdot(cublas, n, d_t, 1, d_s, 1, &t_dot_s));
        omega = (t_dot_t > 1e-300) ? t_dot_s / t_dot_t : 0.0;

        // x = x + alpha*p + omega*s
        CUBLAS_CHECK(cublasDaxpy(cublas, n, &alpha_bc, d_p, 1, d_x, 1));
        CUBLAS_CHECK(cublasDaxpy(cublas, n, &omega, d_s, 1, d_x, 1));

        // r = s - omega*t
        CUDA_CHECK(cudaMemcpy(d_r, d_s, n * sizeof(double), cudaMemcpyDeviceToDevice));
        double neg_omg = -omega;
        CUBLAS_CHECK(cublasDaxpy(cublas, n, &neg_omg, d_t, 1, d_r, 1));

        double r_norm;
        CUBLAS_CHECK(cublasDnrm2(cublas, n, d_r, 1, &r_norm));
        result.residual = r_norm / b_norm;
        result.iterations = iter + 1;

        if (r_norm / b_norm < tol) {
            result.converged = true;
            break;
        }

        rho_old = rho_new;
    }

    // Copy solution back to host
    result.x.resize(n);
    CUDA_CHECK(cudaMemcpy(result.x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    auto t_end = std::chrono::high_resolution_clock::now();
    result.gpu_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Cleanup
    cusparseDestroySpMat(matA);
    cudaFree(buffer);
    cudaFree(d_vals); cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_r); cudaFree(d_r_hat);
    cudaFree(d_p); cudaFree(d_v); cudaFree(d_s); cudaFree(d_t);
    cusparseDestroy(cusparse);
    cublasDestroy(cublas);

    return result;
}

bool detect_gpu() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

std::string gpu_name() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        return std::string(prop.name);
    }
    return "unknown";
}

} // namespace twofluid

#endif
