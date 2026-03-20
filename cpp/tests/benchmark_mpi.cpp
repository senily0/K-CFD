/**
 * MPI Performance Benchmark
 *
 * Large-scale 3D channel flow for scaling test.
 * Mesh: configurable (default 100x50x50 = 250K cells)
 *
 * Usage:
 *   Serial:   benchmark_mpi [nx] [ny] [nz] [max_iter]
 *   Parallel: mpiexec -n 4 benchmark_mpi [nx] [ny] [nz] [max_iter]
 *
 * Hardware target: Intel Core Ultra 5 125H (14C/18T), 32GB RAM
 * Recommended sizes:
 *   Small:  100x50x50  = 250K cells (~1GB)
 *   Medium: 200x80x80  = 1.28M cells (~4GB)
 *   Large:  200x100x100 = 2M cells (~8GB)
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "twofluid/mesh.hpp"
#include "twofluid/mesh_generator_3d.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/simple_solver.hpp"
#include "twofluid/vtk_writer.hpp"
#include "twofluid/mpi_comm.hpp"
#include "twofluid/partitioning.hpp"

using namespace twofluid;

int main(int argc, char* argv[]) {
    // Parse arguments
    int nx = (argc > 1) ? std::atoi(argv[1]) : 100;
    int ny = (argc > 2) ? std::atoi(argv[2]) : 50;
    int nz = (argc > 3) ? std::atoi(argv[3]) : 50;
    int max_iter = (argc > 4) ? std::atoi(argv[4]) : 100;

    MPIComm comm;
    int n_cells_total = nx * ny * nz;

    if (comm.is_root()) {
        std::cout << "================================================================\n";
        std::cout << "  K-CFD MPI Performance Benchmark\n";
        std::cout << "================================================================\n";
        std::cout << "  MPI ranks:    " << comm.size() << "\n";
        std::cout << "  Mesh:         " << nx << " x " << ny << " x " << nz
                  << " = " << n_cells_total << " cells\n";
        std::cout << "  Max iter:     " << max_iter << "\n";
        std::cout << "  Est. memory:  ~" << (n_cells_total * 300) / (1024*1024) << " MB\n";
        std::cout << "================================================================\n\n";
    }

    // ============================================================
    // Phase 1: Mesh generation (root only for large meshes)
    // ============================================================
    auto t_mesh_start = std::chrono::high_resolution_clock::now();

    double Lx = 10.0, Ly = 1.0, Lz = 1.0;
    double Re = 100.0;
    double U_mean = 1.0;
    double rho = 1.0;
    double mu = rho * U_mean * Ly / Re;

    FVMesh mesh(3);

    if (comm.is_root()) {
        std::cout << "Generating 3D mesh..." << std::flush;
    }

    mesh = generate_3d_channel_mesh(Lx, Ly, Lz, nx, ny, nz);

    auto t_mesh_end = std::chrono::high_resolution_clock::now();
    double mesh_time = std::chrono::duration<double>(t_mesh_end - t_mesh_start).count();

    if (comm.is_root()) {
        std::cout << " done (" << std::fixed << std::setprecision(2)
                  << mesh_time << " s)\n";
        std::cout << "  Cells: " << mesh.n_cells
                  << "  Faces: " << mesh.n_faces
                  << " (" << mesh.n_internal_faces << " internal)\n";

        // Mesh quality
        auto quality = mesh.compute_quality();
        std::cout << "  Quality: non-orth=" << std::setprecision(1)
                  << quality.max_non_orthogonality << "°"
                  << "  skewness=" << std::setprecision(3)
                  << quality.max_skewness
                  << "  aspect=" << std::setprecision(1)
                  << quality.max_aspect_ratio << "\n\n";
    }

    // ============================================================
    // Phase 2: Domain decomposition (if parallel)
    // ============================================================
    double decomp_time = 0.0;
    std::vector<int> part_ids;

    if (comm.is_parallel()) {
        auto t_dec_start = std::chrono::high_resolution_clock::now();

        if (comm.is_root()) {
            std::cout << "Partitioning into " << comm.size() << " domains..." << std::flush;
            part_ids = RCBPartitioner::partition(mesh, comm.size());
        }

        // In real MPI, broadcast part_ids here
        // For serial benchmark simulation, we just measure partition time

        auto t_dec_end = std::chrono::high_resolution_clock::now();
        decomp_time = std::chrono::duration<double>(t_dec_end - t_dec_start).count();

        if (comm.is_root()) {
            // Count cells per partition
            std::vector<int> counts(comm.size(), 0);
            for (int pid : part_ids) counts[pid]++;
            std::cout << " done (" << std::setprecision(2) << decomp_time << " s)\n";
            std::cout << "  Cells per rank: ";
            for (int r = 0; r < comm.size(); ++r) {
                std::cout << counts[r];
                if (r < comm.size() - 1) std::cout << " / ";
            }
            std::cout << "\n\n";
        }
    }

    // ============================================================
    // Phase 3: Serial solve (baseline)
    // ============================================================
    if (comm.is_root()) {
        std::cout << "Solving (serial, " << max_iter << " iterations)..." << std::flush;
    }

    auto t_solve_start = std::chrono::high_resolution_clock::now();

    SIMPLESolver solver(mesh, rho, mu);
    solver.max_iter = max_iter;
    solver.tol = 1e-6;  // tight tol so we always run max_iter
    solver.alpha_u = 0.7;
    solver.alpha_p = 0.3;
    solver.linear_solver_type = "bicgstab";  // essential for large 3D

    // BCs: Poiseuille inlet, zero-grad outlet, walls
    int n_inlet = static_cast<int>(mesh.boundary_patches["inlet"].size());
    Eigen::MatrixXd inlet_U(n_inlet, 3);
    for (int j = 0; j < n_inlet; ++j) {
        int fid = mesh.boundary_patches["inlet"][j];
        double y = mesh.faces[fid].center[1];
        double z = mesh.faces[fid].center[2];
        // Simplified parabolic: u = U * 4 * y/Ly * (1-y/Ly) * 4 * z/Lz * (1-z/Lz)
        double u_prof = 2.25 * U_mean * 4.0 * (y/Ly) * (1.0 - y/Ly)
                        * 4.0 * (z/Lz) * (1.0 - z/Lz);
        inlet_U(j, 0) = u_prof;
        inlet_U(j, 1) = 0.0;
        inlet_U(j, 2) = 0.0;
    }
    solver.set_inlet("inlet", inlet_U);
    solver.set_outlet("outlet", 0.0);
    solver.set_wall("wall_bottom");
    solver.set_wall("wall_top");
    solver.set_wall("wall_front");
    solver.set_wall("wall_back");

    auto result = solver.solve_steady();

    auto t_solve_end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(t_solve_end - t_solve_start).count();

    if (comm.is_root()) {
        std::cout << " done\n\n";

        // Results
        double u_max = solver.velocity().magnitude().maxCoeff();
        double p_range = solver.pressure().max() - solver.pressure().min();

        std::cout << "================================================================\n";
        std::cout << "  RESULTS\n";
        std::cout << "================================================================\n";
        std::cout << "  Converged:      " << (result.converged ? "yes" : "no") << "\n";
        std::cout << "  Iterations:     " << result.iterations << "\n";
        std::cout << "  Final residual: " << std::scientific << std::setprecision(3);
        if (!result.residuals.empty()) std::cout << result.residuals.back();
        else std::cout << "N/A";
        std::cout << "\n";
        std::cout << "  |U| max:        " << std::fixed << std::setprecision(4) << u_max << "\n";
        std::cout << "  p range:        " << std::setprecision(4) << p_range << "\n\n";

        // Performance
        double cells_per_sec = static_cast<double>(n_cells_total) * result.iterations / solve_time;
        std::cout << "================================================================\n";
        std::cout << "  PERFORMANCE\n";
        std::cout << "================================================================\n";
        std::cout << "  Mesh generation: " << std::setprecision(2) << mesh_time << " s\n";
        if (decomp_time > 0)
            std::cout << "  Decomposition:   " << decomp_time << " s\n";
        std::cout << "  Solve time:      " << solve_time << " s\n";
        std::cout << "  Time/iteration:  " << std::setprecision(4)
                  << solve_time / result.iterations << " s\n";
        std::cout << "  Throughput:      " << std::scientific << std::setprecision(2)
                  << cells_per_sec << " cell-iters/s\n";
        std::cout << "  Memory est:      " << std::fixed << std::setprecision(0)
                  << (n_cells_total * 300.0) / (1024*1024) << " MB\n";
        std::cout << "================================================================\n";

        // Write VTU output for ParaView visualization
        {
            std::unordered_map<std::string, Eigen::VectorXd> scalar_data;
            std::unordered_map<std::string, Eigen::MatrixXd> vector_data;
            scalar_data["pressure"] = solver.pressure().values;
            scalar_data["velocity_magnitude"] = solver.velocity().magnitude();
            vector_data["velocity"] = solver.velocity().values;
            std::string vtu_file = "benchmark_" + std::to_string(nx) + "x"
                                 + std::to_string(ny) + "x" + std::to_string(nz) + ".vtu";
            try {
                write_vtu(vtu_file, mesh, scalar_data, vector_data);
                std::cout << "\n  VTU output: " << vtu_file << "\n";
            } catch (...) {
                std::cout << "\n  VTU write failed\n";
            }
        }

        // Scaling projection
        if (comm.size() == 1) {
            std::cout << "\n  Projected scaling (ideal):\n";
            for (int np : {2, 4, 8, 14}) {
                std::cout << "    " << np << " ranks: "
                          << std::setprecision(2) << solve_time / np << " s "
                          << "(" << std::setprecision(1) << np * 100.0 / 1.0 << "% ideal)\n";
            }
        }
    }

    return 0;
}
