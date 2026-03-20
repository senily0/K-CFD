/**
 * benchmark_gpu.cu -- Standalone CUDA GPU vs CPU BiCGSTAB benchmark
 *
 * Builds and runs independently with nvcc (no Eigen dependency):
 *   nvcc -O2 -o benchmark_gpu.exe benchmark_gpu.cu -lcusparse -lcublas
 *
 * Generates a 3D Laplacian on a structured grid, solves with:
 *   1. CPU BiCGSTAB (plain C++, no libraries)
 *   2. GPU BiCGSTAB (cuSPARSE + cuBLAS)
 * Reports: iterations, residual, wall time, speedup.
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>

// ============================================================
// 3D Laplacian CSR builder (7-point stencil + diagonal shift)
// ============================================================
struct CSRMatrix {
    int n;
    int nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> vals;
};

static CSRMatrix build_3d_laplacian(int nx, int ny, int nz, double diag_shift = 0.001) {
    int n = nx * ny * nz;
    CSRMatrix A;
    A.n = n;
    A.row_ptr.resize(n + 1);

    // First pass: count non-zeros per row
    A.row_ptr[0] = 0;
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        int row = i + j * nx + k * nx * ny;
        int count = 1; // diagonal
        if (i > 0) ++count;
        if (i < nx - 1) ++count;
        if (j > 0) ++count;
        if (j < ny - 1) ++count;
        if (k > 0) ++count;
        if (k < nz - 1) ++count;
        A.row_ptr[row + 1] = A.row_ptr[row] + count;
    }
    A.nnz = A.row_ptr[n];
    A.col_idx.resize(A.nnz);
    A.vals.resize(A.nnz);

    // Second pass: fill entries (sorted column order)
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        int row = i + j * nx + k * nx * ny;
        int idx = A.row_ptr[row];
        double diag = diag_shift;

        // -z neighbor
        if (k > 0) {
            A.col_idx[idx] = row - nx * ny;
            A.vals[idx] = -1.0;
            diag += 1.0;
            ++idx;
        }
        // -y neighbor
        if (j > 0) {
            A.col_idx[idx] = row - nx;
            A.vals[idx] = -1.0;
            diag += 1.0;
            ++idx;
        }
        // -x neighbor
        if (i > 0) {
            A.col_idx[idx] = row - 1;
            A.vals[idx] = -1.0;
            diag += 1.0;
            ++idx;
        }
        // diagonal
        // (will be filled after counting off-diag contributions)
        int diag_idx = idx;
        A.col_idx[idx] = row;
        A.vals[idx] = 0.0;  // placeholder
        ++idx;
        // +x neighbor
        if (i < nx - 1) {
            A.col_idx[idx] = row + 1;
            A.vals[idx] = -1.0;
            diag += 1.0;
            ++idx;
        }
        // +y neighbor
        if (j < ny - 1) {
            A.col_idx[idx] = row + nx;
            A.vals[idx] = -1.0;
            diag += 1.0;
            ++idx;
        }
        // +z neighbor
        if (k < nz - 1) {
            A.col_idx[idx] = row + nx * ny;
            A.vals[idx] = -1.0;
            diag += 1.0;
            ++idx;
        }
        A.vals[diag_idx] = diag;
    }

    return A;
}

// ============================================================
// CPU BiCGSTAB (self-contained, no external library)
// ============================================================
static void cpu_spmv(const CSRMatrix& A, const double* x, double* y) {
    for (int i = 0; i < A.n; ++i) {
        double sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.vals[j] * x[A.col_idx[j]];
        }
        y[i] = sum;
    }
}

static double cpu_dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static double cpu_nrm2(const double* a, int n) {
    return std::sqrt(cpu_dot(a, a, n));
}

static void cpu_axpy(double alpha, const double* x, double* y, int n) {
    for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

static void cpu_scal(double alpha, double* x, int n) {
    for (int i = 0; i < n; ++i) x[i] *= alpha;
}

static void cpu_copy(const double* src, double* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = src[i];
}

struct SolveResult {
    bool converged;
    int iterations;
    double residual;
    double time_ms;
};

static SolveResult cpu_bicgstab(const CSRMatrix& A, const double* rhs,
                                 double* x, double tol, int maxiter) {
    int n = A.n;
    std::vector<double> r(n), r_hat(n), p(n), v(n), s(n), t(n), tmp(n);

    auto t0 = std::chrono::high_resolution_clock::now();

    // r = b - A*x
    cpu_spmv(A, x, r.data());
    for (int i = 0; i < n; ++i) r[i] = rhs[i] - r[i];

    cpu_copy(r.data(), r_hat.data(), n);
    for (int i = 0; i < n; ++i) { p[i] = 0.0; v[i] = 0.0; }

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    double b_norm = cpu_nrm2(rhs, n);
    if (b_norm < 1e-30) b_norm = 1.0;

    SolveResult res = {false, 0, 1.0, 0.0};

    for (int iter = 0; iter < maxiter; ++iter) {
        double rho_new = cpu_dot(r_hat.data(), r.data(), n);
        if (std::abs(rho_new) < 1e-300) break;

        double beta = (rho_new / rho_old) * (alpha / omega);

        // p = r + beta*(p - omega*v)
        cpu_axpy(-omega, v.data(), p.data(), n);
        cpu_scal(beta, p.data(), n);
        cpu_axpy(1.0, r.data(), p.data(), n);

        // v = A*p
        cpu_spmv(A, p.data(), v.data());

        double denom = cpu_dot(r_hat.data(), v.data(), n);
        if (std::abs(denom) < 1e-300) break;
        alpha = rho_new / denom;

        // s = r - alpha*v
        cpu_copy(r.data(), s.data(), n);
        cpu_axpy(-alpha, v.data(), s.data(), n);

        double s_norm = cpu_nrm2(s.data(), n);
        if (s_norm / b_norm < tol) {
            cpu_axpy(alpha, p.data(), x, n);
            res.iterations = iter + 1;
            res.residual = s_norm / b_norm;
            res.converged = true;
            break;
        }

        // t = A*s
        cpu_spmv(A, s.data(), t.data());

        double t_dot_t = cpu_dot(t.data(), t.data(), n);
        double t_dot_s = cpu_dot(t.data(), s.data(), n);
        omega = (t_dot_t > 1e-300) ? t_dot_s / t_dot_t : 0.0;

        // x += alpha*p + omega*s
        cpu_axpy(alpha, p.data(), x, n);
        cpu_axpy(omega, s.data(), x, n);

        // r = s - omega*t
        cpu_copy(s.data(), r.data(), n);
        cpu_axpy(-omega, t.data(), r.data(), n);

        double r_norm = cpu_nrm2(r.data(), n);
        res.residual = r_norm / b_norm;
        res.iterations = iter + 1;

        if (r_norm / b_norm < tol) {
            res.converged = true;
            break;
        }

        rho_old = rho_new;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

// ============================================================
// GPU BiCGSTAB (cuSPARSE + cuBLAS)
// ============================================================
static SolveResult gpu_bicgstab(const CSRMatrix& A, const double* rhs,
                                 double* x, double tol, int maxiter) {
    int n = A.n;
    int nnz = A.nnz;
    SolveResult res = {false, 0, 1.0, 0.0};

    auto t_start = std::chrono::high_resolution_clock::now();

    double *d_vals, *d_x, *d_b, *d_r, *d_r_hat, *d_p, *d_v, *d_s, *d_t;
    int *d_row_ptr, *d_col_idx;

    cudaMalloc(&d_vals, nnz * sizeof(double));
    cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_b, n * sizeof(double));
    cudaMalloc(&d_r, n * sizeof(double));
    cudaMalloc(&d_r_hat, n * sizeof(double));
    cudaMalloc(&d_p, n * sizeof(double));
    cudaMalloc(&d_v, n * sizeof(double));
    cudaMalloc(&d_s, n * sizeof(double));
    cudaMalloc(&d_t, n * sizeof(double));

    cudaMemcpy(d_vals, A.vals.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, A.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, A.col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, rhs, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);

    cusparseHandle_t cusparse;
    cublasHandle_t cublas;
    cusparseCreate(&cusparse);
    cublasCreate(&cublas);

    cusparseSpMatDescr_t matDescr;
    cusparseCreateCsr(&matDescr, n, n, nnz,
                       d_row_ptr, d_col_idx, d_vals,
                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    // Allocate SpMV buffer
    double alpha_sp = 1.0, beta_sp = 0.0;
    cusparseDnVecDescr_t vecBufIn, vecBufOut;
    cusparseCreateDnVec(&vecBufIn, n, d_p, CUDA_R_64F);
    cusparseCreateDnVec(&vecBufOut, n, d_v, CUDA_R_64F);
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha_sp, matDescr, vecBufIn, &beta_sp, vecBufOut,
                             CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    void* buffer;
    cudaMalloc(&buffer, bufferSize);
    cusparseDestroyDnVec(vecBufIn);
    cusparseDestroyDnVec(vecBufOut);

    // SpMV helper
    auto spmv = [&](double* in, double* out) {
        cusparseDnVecDescr_t vIn, vOut;
        cusparseCreateDnVec(&vIn, n, in, CUDA_R_64F);
        cusparseCreateDnVec(&vOut, n, out, CUDA_R_64F);
        double a = 1.0, b = 0.0;
        cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &a, matDescr, vIn, &b, vOut,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
        cusparseDestroyDnVec(vIn);
        cusparseDestroyDnVec(vOut);
    };

    // r = b - A*x0
    spmv(d_x, d_r);
    double neg_one = -1.0, one = 1.0;
    cublasDscal(cublas, n, &neg_one, d_r, 1);
    cublasDaxpy(cublas, n, &one, d_b, 1, d_r, 1);

    cudaMemcpy(d_r_hat, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemset(d_p, 0, n * sizeof(double));
    cudaMemset(d_v, 0, n * sizeof(double));

    double rho_old = 1.0, alpha_bc = 1.0, omega = 1.0;
    double b_norm;
    cublasDnrm2(cublas, n, d_b, 1, &b_norm);
    if (b_norm < 1e-30) b_norm = 1.0;

    for (int iter = 0; iter < maxiter; ++iter) {
        double rho_new;
        cublasDdot(cublas, n, d_r_hat, 1, d_r, 1, &rho_new);
        if (std::abs(rho_new) < 1e-300) break;

        double beta = (rho_new / rho_old) * (alpha_bc / omega);

        double neg_omega = -omega;
        cublasDaxpy(cublas, n, &neg_omega, d_v, 1, d_p, 1);
        cublasDscal(cublas, n, &beta, d_p, 1);
        cublasDaxpy(cublas, n, &one, d_r, 1, d_p, 1);

        spmv(d_p, d_v);

        double denom;
        cublasDdot(cublas, n, d_r_hat, 1, d_v, 1, &denom);
        if (std::abs(denom) < 1e-300) break;
        alpha_bc = rho_new / denom;

        cudaMemcpy(d_s, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice);
        double neg_alpha = -alpha_bc;
        cublasDaxpy(cublas, n, &neg_alpha, d_v, 1, d_s, 1);

        double s_norm;
        cublasDnrm2(cublas, n, d_s, 1, &s_norm);
        if (s_norm / b_norm < tol) {
            cublasDaxpy(cublas, n, &alpha_bc, d_p, 1, d_x, 1);
            res.iterations = iter + 1;
            res.residual = s_norm / b_norm;
            res.converged = true;
            break;
        }

        spmv(d_s, d_t);

        double t_dot_t, t_dot_s;
        cublasDdot(cublas, n, d_t, 1, d_t, 1, &t_dot_t);
        cublasDdot(cublas, n, d_t, 1, d_s, 1, &t_dot_s);
        omega = (t_dot_t > 1e-300) ? t_dot_s / t_dot_t : 0.0;

        cublasDaxpy(cublas, n, &alpha_bc, d_p, 1, d_x, 1);
        cublasDaxpy(cublas, n, &omega, d_s, 1, d_x, 1);

        cudaMemcpy(d_r, d_s, n * sizeof(double), cudaMemcpyDeviceToDevice);
        double neg_omg = -omega;
        cublasDaxpy(cublas, n, &neg_omg, d_t, 1, d_r, 1);

        double r_norm;
        cublasDnrm2(cublas, n, d_r, 1, &r_norm);
        res.residual = r_norm / b_norm;
        res.iterations = iter + 1;

        if (r_norm / b_norm < tol) {
            res.converged = true;
            break;
        }
        rho_old = rho_new;
    }

    cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    auto t_end = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    cusparseDestroySpMat(matDescr);
    cudaFree(buffer);
    cudaFree(d_vals); cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_r); cudaFree(d_r_hat);
    cudaFree(d_p); cudaFree(d_v); cudaFree(d_s); cudaFree(d_t);
    cusparseDestroy(cusparse);
    cublasDestroy(cublas);

    return res;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    // Grid size: default 40x40x40 = 64000 cells
    int nx = 40, ny = 40, nz = 40;
    if (argc >= 4) {
        nx = std::atoi(argv[1]);
        ny = std::atoi(argv[2]);
        nz = std::atoi(argv[3]);
    }
    int n = nx * ny * nz;
    double tol = 1e-8;
    int maxiter = 5000;

    printf("================================================================\n");
    printf("  GPU vs CPU BiCGSTAB Benchmark\n");
    printf("================================================================\n");

    // Print GPU info
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("  GPU: %s (%.0f MB, SM %d.%d)\n",
               prop.name, prop.totalGlobalMem / 1048576.0,
               prop.major, prop.minor);
    } else {
        printf("  No CUDA GPU detected!\n");
        return 1;
    }

    printf("  Grid: %dx%dx%d = %d cells\n", nx, ny, nz, n);
    printf("  Solver: BiCGSTAB, tol=%.1e, maxiter=%d\n\n", tol, maxiter);

    // Build system
    printf("  Building 3D Laplacian (7-point stencil)...\n");
    auto A = build_3d_laplacian(nx, ny, nz, 0.001);
    printf("  Matrix: %d x %d, nnz = %d\n", n, n, A.nnz);

    // Build RHS: f(x,y,z) = sin(pi*x/nx)*cos(pi*y/ny)*sin(pi*z/nz)
    std::vector<double> rhs(n);
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        int idx = i + j * nx + k * nx * ny;
        double x = static_cast<double>(i) / nx;
        double y = static_cast<double>(j) / ny;
        double z = static_cast<double>(k) / nz;
        rhs[idx] = std::sin(M_PI * x) * std::cos(M_PI * y) * std::sin(M_PI * z);
    }

    // --- CPU solve ---
    printf("\n  [CPU BiCGSTAB]\n");
    std::vector<double> x_cpu(n, 0.0);
    auto cpu_res = cpu_bicgstab(A, rhs.data(), x_cpu.data(), tol, maxiter);
    printf("    converged: %s\n", cpu_res.converged ? "yes" : "NO");
    printf("    iterations: %d\n", cpu_res.iterations);
    printf("    residual: %.3e\n", cpu_res.residual);
    printf("    time: %.1f ms\n", cpu_res.time_ms);

    // --- GPU solve ---
    printf("\n  [GPU BiCGSTAB (cuSPARSE + cuBLAS)]\n");
    std::vector<double> x_gpu(n, 0.0);
    auto gpu_res = gpu_bicgstab(A, rhs.data(), x_gpu.data(), tol, maxiter);
    printf("    converged: %s\n", gpu_res.converged ? "yes" : "NO");
    printf("    iterations: %d\n", gpu_res.iterations);
    printf("    residual: %.3e\n", gpu_res.residual);
    printf("    time: %.1f ms (incl. transfer)\n", gpu_res.time_ms);

    // --- Comparison ---
    printf("\n  [Comparison]\n");
    if (cpu_res.time_ms > 0 && gpu_res.time_ms > 0) {
        double speedup = cpu_res.time_ms / gpu_res.time_ms;
        printf("    speedup: %.2fx\n", speedup);
        if (speedup > 1.0) {
            printf("    GPU is %.1f%% faster\n", (speedup - 1.0) * 100.0);
        } else {
            printf("    CPU is %.1f%% faster (GPU overhead dominates at this size)\n",
                   (1.0 / speedup - 1.0) * 100.0);
        }
    }

    // Check solutions agree
    if (cpu_res.converged && gpu_res.converged) {
        double diff = 0.0, norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = x_cpu[i] - x_gpu[i];
            diff += d * d;
            norm += x_cpu[i] * x_cpu[i];
        }
        diff = std::sqrt(diff);
        norm = std::sqrt(norm);
        double rel_diff = (norm > 1e-30) ? diff / norm : diff;
        printf("    solution difference (L2 relative): %.3e\n", rel_diff);
        printf("    solutions %s\n", (rel_diff < 1e-4) ? "AGREE" : "DIFFER");
    }

    // CSV output for parsing
    printf("\n--- CSV ---\n");
    printf("solver,converged,iterations,residual,time_ms\n");
    printf("cpu,%d,%d,%.10e,%.3f\n",
           cpu_res.converged, cpu_res.iterations, cpu_res.residual, cpu_res.time_ms);
    printf("gpu,%d,%d,%.10e,%.3f\n",
           gpu_res.converged, gpu_res.iterations, gpu_res.residual, gpu_res.time_ms);

    printf("\n================================================================\n");
    return (cpu_res.converged && gpu_res.converged) ? 0 : 1;
}
