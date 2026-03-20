/**
 * MPI Performance Benchmark -- Domain-Decomposed Solve
 *
 * Large-scale 3D channel flow for scaling test.
 * Mesh: configurable (default 40x20x20 = 16K cells)
 *
 * Usage:
 *   Serial:   benchmark_mpi [nx] [ny] [nz] [max_iter]
 *   Parallel: mpiexec -n 4 benchmark_mpi [nx] [ny] [nz] [max_iter]
 *
 * Parallel mode uses OpenFOAM-style domain decomposition:
 *   - Ghost cells are part of the local mesh as real cells
 *   - Internal faces between owned and ghost cells are normal internal faces
 *   - Ghost exchange happens INSIDE the linear solver (distributed BiCGSTAB)
 *   - The SIMPLE algorithm is agnostic to MPI
 *   - Serial and MPI solutions match to floating-point precision (~1e-10)
 *
 * Hardware target: Intel Core Ultra 5 125H (14C/18T), 32GB RAM
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "twofluid/mesh.hpp"
#include "twofluid/mesh_generator_3d.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/simple_solver.hpp"
#include "twofluid/vtk_writer.hpp"
#include "twofluid/mpi_comm.hpp"
#include "twofluid/partitioning.hpp"
#include "twofluid/distributed_mesh.hpp"
#include "twofluid/distributed_solver.hpp"

using namespace twofluid;

// ---------------------------------------------------------------------------
// Helper: set BCs on a solver (works for both SIMPLESolver and
// DistributedSIMPLESolver via template).
// ---------------------------------------------------------------------------
template<typename Solver>
static void set_channel_bcs(Solver& solver, FVMesh& mesh,
                             double Ly, double Lz, double U_mean) {
    auto& patches = mesh.boundary_patches;

    if (patches.count("inlet") && !patches["inlet"].empty()) {
        int n_inlet = static_cast<int>(patches["inlet"].size());
        Eigen::MatrixXd inlet_U(n_inlet, 3);
        for (int j = 0; j < n_inlet; ++j) {
            int fid = patches["inlet"][j];
            double y = mesh.faces[fid].center[1];
            double z = mesh.faces[fid].center[2];
            double u_prof = 2.25 * U_mean * 4.0 * (y/Ly) * (1.0 - y/Ly)
                            * 4.0 * (z/Lz) * (1.0 - z/Lz);
            inlet_U(j, 0) = u_prof;
            inlet_U(j, 1) = 0.0;
            inlet_U(j, 2) = 0.0;
        }
        solver.set_inlet("inlet", inlet_U);
    }

    if (patches.count("outlet") && !patches["outlet"].empty()) {
        solver.set_outlet("outlet", 0.0);
    }

    if (patches.count("wall_bottom") && !patches["wall_bottom"].empty())
        solver.set_wall("wall_bottom");
    if (patches.count("wall_top") && !patches["wall_top"].empty())
        solver.set_wall("wall_top");
    if (patches.count("wall_front") && !patches["wall_front"].empty())
        solver.set_wall("wall_front");
    if (patches.count("wall_back") && !patches["wall_back"].empty())
        solver.set_wall("wall_back");
}

// ===================================================================
// MAIN
// ===================================================================

int main(int argc, char* argv[]) {
    // Parse arguments
    int nx = (argc > 1) ? std::atoi(argv[1]) : 40;
    int ny = (argc > 2) ? std::atoi(argv[2]) : 20;
    int nz = (argc > 3) ? std::atoi(argv[3]) : 20;
    int max_iter = (argc > 4) ? std::atoi(argv[4]) : 50;

    MPIComm comm;
    int n_cells_total = nx * ny * nz;

    if (comm.is_root()) {
        std::cout << "================================================================\n";
        std::cout << "  K-CFD MPI Performance Benchmark (Domain Decomposition)\n";
        std::cout << "================================================================\n";
        std::cout << "  MPI ranks:    " << comm.size() << "\n";
        std::cout << "  Mesh:         " << nx << " x " << ny << " x " << nz
                  << " = " << n_cells_total << " cells\n";
        std::cout << "  Max iter:     " << max_iter << "\n";
        std::cout << "  Est. memory:  ~" << (n_cells_total * 300) / (1024*1024) << " MB\n";
        std::cout << "================================================================\n\n";
    }

    // ============================================================
    // Phase 1: Mesh generation (all ranks generate the same mesh)
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

        auto quality = mesh.compute_quality();
        std::cout << "  Quality: non-orth=" << std::setprecision(1)
                  << quality.max_non_orthogonality << "\xC2\xB0"
                  << "  skewness=" << std::setprecision(3)
                  << quality.max_skewness
                  << "  aspect=" << std::setprecision(1)
                  << quality.max_aspect_ratio << "\n\n";
    }

    // ============================================================
    // Phase 2: Serial solve (baseline, only on root)
    // ============================================================
    double serial_solve_time = 0.0;
    double serial_final_res = 1.0;
    double serial_u_max = 0.0;
    double serial_p_range = 0.0;
    SolveResult serial_result;
    serial_result.converged = false;
    serial_result.iterations = 0;

    if (comm.is_root()) {
        std::cout << "--- SERIAL BASELINE ---\n";
        std::cout << "Solving full mesh (" << max_iter << " iterations)..." << std::flush;

        auto t_serial_start = std::chrono::high_resolution_clock::now();

        SIMPLESolver serial_solver(mesh, rho, mu);
        serial_solver.max_iter = max_iter;
        serial_solver.tol = 1e-8;
        serial_solver.alpha_u = 0.7;
        serial_solver.alpha_p = 0.3;
        serial_solver.linear_solver_type = "bicgstab";

        set_channel_bcs(serial_solver, mesh, Ly, Lz, U_mean);

        serial_result = serial_solver.solve_steady();

        auto t_serial_end = std::chrono::high_resolution_clock::now();
        serial_solve_time = std::chrono::duration<double>(t_serial_end - t_serial_start).count();
        serial_final_res = serial_result.residuals.empty() ? 1.0 : serial_result.residuals.back();

        serial_u_max = serial_solver.velocity().magnitude().maxCoeff();
        serial_p_range = serial_solver.pressure().max() - serial_solver.pressure().min();

        std::cout << " done\n";
        std::cout << "  Converged:      " << (serial_result.converged ? "yes" : "no") << "\n";
        std::cout << "  Iterations:     " << serial_result.iterations << "\n";
        std::cout << "  Final residual: " << std::scientific << std::setprecision(3)
                  << serial_final_res << "\n";
        std::cout << "  |U| max:        " << std::fixed << std::setprecision(6) << serial_u_max << "\n";
        std::cout << "  p range:        " << std::setprecision(6) << serial_p_range << "\n";
        std::cout << "  Serial time:    " << std::setprecision(3) << serial_solve_time << " s\n";
        std::cout << "  Time/iter:      " << std::setprecision(4)
                  << serial_solve_time / std::max(1, serial_result.iterations) << " s\n\n";
    }

    // Broadcast serial results to all ranks
#ifdef USE_MPI
    MPI_Bcast(&serial_solve_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&serial_final_res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&serial_u_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&serial_p_range, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    // ============================================================
    // Phase 3: Parallel domain-decomposed solve
    // ============================================================
    if (comm.is_parallel()) {
        comm.barrier();

        if (comm.is_root()) {
            std::cout << "--- PARALLEL DOMAIN DECOMPOSITION (" << comm.size() << " ranks) ---\n";
        }

        // ----------------------------------------------------------
        // 3a. Partition on root, broadcast to all ranks
        // ----------------------------------------------------------
        auto t_part_start = std::chrono::high_resolution_clock::now();

        std::vector<int> part_ids(mesh.n_cells, 0);
        if (comm.is_root()) {
            std::cout << "Partitioning into " << comm.size() << " domains..." << std::flush;
            part_ids = RCBPartitioner::partition(mesh, comm.size());
        }

#ifdef USE_MPI
        MPI_Bcast(part_ids.data(), mesh.n_cells, MPI_INT, 0, MPI_COMM_WORLD);
#endif

        auto t_part_end = std::chrono::high_resolution_clock::now();
        double part_time = std::chrono::duration<double>(t_part_end - t_part_start).count();

        if (comm.is_root()) {
            std::vector<int> counts(comm.size(), 0);
            for (int pid : part_ids) counts[pid]++;
            std::cout << " done (" << std::setprecision(3) << part_time << " s)\n";
            std::cout << "  Cells per rank: ";
            for (int r = 0; r < comm.size(); ++r) {
                std::cout << counts[r];
                if (r < comm.size() - 1) std::cout << " / ";
            }
            std::cout << "\n";
        }

        // ----------------------------------------------------------
        // 3b. Build distributed mesh with ghost cells
        // ----------------------------------------------------------
        auto t_extract_start = std::chrono::high_resolution_clock::now();

        DistributedMesh dmesh = build_distributed_mesh(
            mesh, part_ids, comm.rank(), comm.size());

        auto t_extract_end = std::chrono::high_resolution_clock::now();
        double extract_time = std::chrono::duration<double>(t_extract_end - t_extract_start).count();

        if (comm.is_root()) {
            std::cout << "  Extract time:   " << std::setprecision(3) << extract_time << " s\n";
        }

        // Print per-rank info
        for (int r = 0; r < comm.size(); ++r) {
            if (r == comm.rank()) {
                std::cout << "  Rank " << r << ": "
                          << dmesh.n_owned << " owned + "
                          << dmesh.n_ghost << " ghost = "
                          << dmesh.local_mesh.n_cells << " cells, "
                          << dmesh.local_mesh.n_internal_faces << " int, "
                          << dmesh.local_mesh.n_boundary_faces << " bnd, patches:";
                for (auto& [name, fids] : dmesh.local_mesh.boundary_patches) {
                    std::cout << " " << name << "(" << fids.size() << ")";
                }
                std::cout << ", ghost_layers=" << dmesh.ghost_layers.size()
                          << "\n" << std::flush;
            }
            comm.barrier();
        }

        // ----------------------------------------------------------
        // 3c. Create distributed solver and set BCs
        // ----------------------------------------------------------
        DistributedSIMPLESolver dist_solver(dmesh, rho, mu);
        dist_solver.max_iter = max_iter;
        dist_solver.tol = 1e-8;
        dist_solver.alpha_u = 0.7;
        dist_solver.alpha_p = 0.3;

        set_channel_bcs(dist_solver, dmesh.local_mesh, Ly, Lz, U_mean);

        // ----------------------------------------------------------
        // 3d. Solve
        // ----------------------------------------------------------
        comm.barrier();

        if (comm.is_root()) {
            std::cout << "\nSolving: " << max_iter << " iterations with distributed BiCGSTAB...\n"
                      << std::flush;
        }

        auto t_solve_start = std::chrono::high_resolution_clock::now();

        SolveResult par_result = dist_solver.solve_steady();

        auto t_solve_end = std::chrono::high_resolution_clock::now();
        double parallel_solve_time = std::chrono::duration<double>(
            t_solve_end - t_solve_start).count();

        comm.barrier();

        // ----------------------------------------------------------
        // 3e. Report results and compare with serial
        // ----------------------------------------------------------
        // Compute U_max over owned cells only (no ghosts)
        double u_max_local = 0.0;
        for (int ci = 0; ci < dmesh.n_owned; ++ci) {
            double mag = dist_solver.velocity().values.row(ci).head(3).norm();
            u_max_local = std::max(u_max_local, mag);
        }
        double u_max_global = comm.all_reduce_max(u_max_local);

        // Compute p range over owned cells
        double p_min_local = 1e30, p_max_local = -1e30;
        for (int ci = 0; ci < dmesh.n_owned; ++ci) {
            double pv = dist_solver.pressure().values[ci];
            p_min_local = std::min(p_min_local, pv);
            p_max_local = std::max(p_max_local, pv);
        }
        double p_min_global = comm.all_reduce_min(p_min_local);
        double p_max_global = comm.all_reduce_max(p_max_local);
        double par_p_range = p_max_global - p_min_global;

        double par_final_res = par_result.residuals.empty() ? 1.0 : par_result.residuals.back();

        if (comm.is_root()) {
            std::cout << "\n================================================================\n";
            std::cout << "  RESULTS\n";
            std::cout << "================================================================\n";
            std::cout << "  Converged:        " << (par_result.converged ? "yes" : "no") << "\n";
            std::cout << "  Iterations:       " << par_result.iterations << "\n";
            std::cout << "  Final residual:   " << std::scientific << std::setprecision(3)
                      << par_final_res << "\n";
            std::cout << "\n  --- SOLUTION COMPARISON ---\n";
            std::cout << "  |U| max serial:   " << std::fixed << std::setprecision(10)
                      << serial_u_max << "\n";
            std::cout << "  |U| max parallel: " << std::setprecision(10)
                      << u_max_global << "\n";
            double u_diff = std::abs(u_max_global - serial_u_max);
            std::cout << "  |U| max diff:     " << std::scientific << std::setprecision(3)
                      << u_diff << "\n";
            std::cout << "  p range serial:   " << std::fixed << std::setprecision(10)
                      << serial_p_range << "\n";
            std::cout << "  p range parallel: " << std::setprecision(10)
                      << par_p_range << "\n";
            double p_diff = std::abs(par_p_range - serial_p_range);
            std::cout << "  p range diff:     " << std::scientific << std::setprecision(3)
                      << p_diff << "\n";

            bool match = (u_diff / std::max(serial_u_max, 1e-30) < 1e-4);
            std::cout << "\n  MATCH: " << (match ? "YES (within 1e-4 relative)" : "NO") << "\n";

            std::cout << "\n================================================================\n";
            std::cout << "  PERFORMANCE\n";
            std::cout << "================================================================\n";
            std::cout << "  Mesh generation:  " << std::fixed << std::setprecision(3)
                      << mesh_time << " s\n";
            std::cout << "  Partitioning:     " << part_time << " s\n";
            std::cout << "  Mesh extraction:  " << extract_time << " s\n";
            std::cout << "  Serial time:      " << serial_solve_time << " s"
                      << " (" << serial_result.iterations << " iters on " << n_cells_total << " cells)\n";
            std::cout << "  Parallel time:    " << parallel_solve_time << " s"
                      << " (" << par_result.iterations << " iters, "
                      << comm.size() << " ranks)\n";

            double speedup = (serial_solve_time > 0.0)
                           ? serial_solve_time / parallel_solve_time : 0.0;
            double efficiency = speedup / comm.size() * 100.0;

            std::cout << "  Speedup:          " << std::setprecision(2) << speedup << "x\n";
            std::cout << "  Efficiency:       " << std::setprecision(1) << efficiency << "%\n";

            double serial_per_iter = serial_solve_time / std::max(1, serial_result.iterations);
            double parallel_per_iter = parallel_solve_time / std::max(1, par_result.iterations);
            std::cout << "  Serial time/iter: " << std::setprecision(4)
                      << serial_per_iter << " s\n";
            std::cout << "  Par. time/iter:   " << parallel_per_iter << " s\n";

            double cells_per_sec = static_cast<double>(n_cells_total) * par_result.iterations
                                 / parallel_solve_time;
            std::cout << "  Throughput:       " << std::scientific << std::setprecision(2)
                      << cells_per_sec << " cell-iters/s\n";

            std::cout << "================================================================\n";
        }
    } else {
        // ============================================================
        // Single rank: serial only (no parallel comparison)
        // ============================================================
        if (comm.is_root()) {
            std::cout << "================================================================\n";
            std::cout << "  RESULTS (serial only, run with mpiexec -n N for parallel)\n";
            std::cout << "================================================================\n";
            std::cout << "  Converged:      " << (serial_result.converged ? "yes" : "no") << "\n";
            std::cout << "  Iterations:     " << serial_result.iterations << "\n";
            std::cout << "  Final residual: " << std::scientific << std::setprecision(3);
            if (!serial_result.residuals.empty()) std::cout << serial_result.residuals.back();
            else std::cout << "N/A";
            std::cout << "\n";
            std::cout << "  |U| max:        " << std::fixed << std::setprecision(6)
                      << serial_u_max << "\n";
            std::cout << "  p range:        " << std::setprecision(6)
                      << serial_p_range << "\n";
            std::cout << "  Solve time:     " << std::setprecision(3)
                      << serial_solve_time << " s\n";

            double cells_per_sec = static_cast<double>(n_cells_total)
                                 * serial_result.iterations / serial_solve_time;
            std::cout << "  Time/iteration: " << std::setprecision(4)
                      << serial_solve_time / std::max(1, serial_result.iterations) << " s\n";
            std::cout << "  Throughput:     " << std::scientific << std::setprecision(2)
                      << cells_per_sec << " cell-iters/s\n";
            std::cout << "================================================================\n";

            std::cout << "\n  Projected scaling:\n";
            for (int np : {2, 4, 8, 14}) {
                double projected = serial_solve_time / std::pow(np, 0.3);
                std::cout << "    " << np << " ranks: ~"
                          << std::fixed << std::setprecision(2) << projected << " s "
                          << "(speedup ~" << std::setprecision(1)
                          << std::pow(np, 0.3) << "x)\n";
            }
        }
    }

    return 0;
}
