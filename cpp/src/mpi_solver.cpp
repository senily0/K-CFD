#include "twofluid/mpi_solver.hpp"
#include <algorithm>
#include <stdexcept>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace twofluid {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MPISIMPLESolver::MPISIMPLESolver(MPIComm& comm,
                                  FVMesh& global_mesh,
                                  double rho,
                                  double mu)
    : comm_(comm) {
    // On root: partition and broadcast part_ids; on other ranks: receive.
    std::vector<int> part_ids;

#ifdef USE_MPI
    if (comm_.is_root()) {
        part_ids = RCBPartitioner::partition(global_mesh, comm_.size());
    } else {
        part_ids.resize(global_mesh.n_cells, 0);
    }
    if (comm_.is_parallel()) {
        MPI_Bcast(part_ids.data(), global_mesh.n_cells, MPI_INT,
                  0, MPI_COMM_WORLD);
    }
#else
    part_ids = RCBPartitioner::partition(global_mesh, 1);
#endif

    local_mesh_   = extract_local_mesh(global_mesh, part_ids, comm_.rank());
    ghost_layers_ = build_ghost_layers(global_mesh, part_ids, comm_.rank());

    local_solver_ = std::make_unique<SIMPLESolver>(local_mesh_.mesh, rho, mu);
}

// ---------------------------------------------------------------------------
// Ghost exchange helpers
// ---------------------------------------------------------------------------

void MPISIMPLESolver::exchange_ghosts(ScalarField& field) {
    if (!comm_.is_parallel()) return;

    // Non-blocking approach: post all sends, then all receives.
    // For simplicity use blocking paired sends/recvs ordered by rank.
    for (auto& layer : ghost_layers_) {
        const int nbr = layer.neighbor_rank;
        if (nbr < comm_.rank()) {
            // Send first, then receive
            comm_.send_scalar(field.values, layer.send_cells, nbr, 100);
            Eigen::VectorXd recv = comm_.recv_scalar(
                static_cast<int>(layer.recv_cells.size()), nbr, 100);
            for (std::size_t i = 0; i < layer.recv_cells.size(); ++i) {
                const int lci = layer.recv_cells[i];
                if (lci < field.values.size()) {
                    field.values[lci] = recv[i];
                }
            }
        } else {
            // Receive first, then send
            Eigen::VectorXd recv = comm_.recv_scalar(
                static_cast<int>(layer.recv_cells.size()), nbr, 100);
            comm_.send_scalar(field.values, layer.send_cells, nbr, 100);
            for (std::size_t i = 0; i < layer.recv_cells.size(); ++i) {
                const int lci = layer.recv_cells[i];
                if (lci < field.values.size()) {
                    field.values[lci] = recv[i];
                }
            }
        }
    }
}

void MPISIMPLESolver::exchange_ghosts(VectorField& field) {
    if (!comm_.is_parallel()) return;

    const int ndim = local_mesh_.mesh.ndim;
    for (auto& layer : ghost_layers_) {
        const int nbr = layer.neighbor_rank;
        if (nbr < comm_.rank()) {
            comm_.send_vector(field.values, layer.send_cells, nbr, 101);
            Eigen::MatrixXd recv = comm_.recv_vector(
                static_cast<int>(layer.recv_cells.size()), ndim, nbr, 101);
            for (std::size_t i = 0; i < layer.recv_cells.size(); ++i) {
                const int lci = layer.recv_cells[i];
                if (lci < field.values.rows()) {
                    field.values.row(lci) = recv.row(i);
                }
            }
        } else {
            Eigen::MatrixXd recv = comm_.recv_vector(
                static_cast<int>(layer.recv_cells.size()), ndim, nbr, 101);
            comm_.send_vector(field.values, layer.send_cells, nbr, 101);
            for (std::size_t i = 0; i < layer.recv_cells.size(); ++i) {
                const int lci = layer.recv_cells[i];
                if (lci < field.values.rows()) {
                    field.values.row(lci) = recv.row(i);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Global residual
// ---------------------------------------------------------------------------

double MPISIMPLESolver::global_residual(double local_res) {
    return comm_.all_reduce_max(local_res);
}

// ---------------------------------------------------------------------------
// solve_steady
// ---------------------------------------------------------------------------

SolveResult MPISIMPLESolver::solve_steady() {
    // Delegate the actual per-iteration logic to the local solver.
    // We override convergence checking with a global all-reduce.
    // The local solver's solve_steady() runs all iterations; instead we
    // replicate a simplified outer loop here.

    SolveResult result;
    result.converged  = false;
    result.iterations = 0;

    const int    max_iter = local_solver_->max_iter;
    const double tol      = local_solver_->tol;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Exchange ghost values for velocity and pressure before each iteration
        exchange_ghosts(local_solver_->velocity());
        exchange_ghosts(local_solver_->pressure());

        // Run a single SIMPLE iteration via the local solver's transient step
        // with a large dt to mimic steady-state behaviour.
        SolveResult step = local_solver_->solve_transient_step(1e10);

        // Compute global residual
        const double local_res  = step.residuals.empty() ? 0.0 : step.residuals.back();
        const double global_res = global_residual(local_res);

        result.residuals.push_back(global_res);
        result.iterations = iter + 1;

        if (global_res < tol) {
            result.converged = true;
            break;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Gather helpers
// ---------------------------------------------------------------------------

Eigen::VectorXd MPISIMPLESolver::gather_scalar(const ScalarField& local_field) {
    const int local_n = local_mesh_.mesh.n_cells;
    Eigen::VectorXd global_result;

#ifdef USE_MPI
    // Collect the number of local cells from each rank on root
    std::vector<int> recv_counts(comm_.size(), 0);
    MPI_Gather(&local_n, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<int> displs(comm_.size(), 0);
    int total = 0;
    if (comm_.is_root()) {
        for (int r = 0; r < comm_.size(); ++r) {
            displs[r] = total;
            total += recv_counts[r];
        }
        global_result.resize(total);
    }

    // Copy local owned values (not ghost) into send buffer
    std::vector<double> send_buf(local_n);
    for (int i = 0; i < local_n; ++i) send_buf[i] = local_field.values[i];

    MPI_Gatherv(send_buf.data(), local_n, MPI_DOUBLE,
                global_result.data(), recv_counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    global_result = local_field.values;
#endif
    return global_result;
}

Eigen::MatrixXd MPISIMPLESolver::gather_vector(const VectorField& local_field) {
    const int local_n = local_mesh_.mesh.n_cells;
    const int ndim    = local_mesh_.mesh.ndim;
    Eigen::MatrixXd global_result;

#ifdef USE_MPI
    // Pack local rows into flat buffer
    std::vector<double> send_buf(local_n * ndim);
    for (int i = 0; i < local_n; ++i) {
        for (int d = 0; d < ndim; ++d) {
            send_buf[i * ndim + d] = local_field.values(i, d);
        }
    }

    const int send_count = local_n * ndim;
    std::vector<int> recv_counts(comm_.size(), 0);
    MPI_Gather(&send_count, 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<int> displs(comm_.size(), 0);
    int total_values = 0;
    if (comm_.is_root()) {
        for (int r = 0; r < comm_.size(); ++r) {
            displs[r] = total_values;
            total_values += recv_counts[r];
        }
    }

    std::vector<double> recv_buf;
    if (comm_.is_root()) recv_buf.resize(total_values);

    MPI_Gatherv(send_buf.data(), send_count, MPI_DOUBLE,
                recv_buf.data(), recv_counts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (comm_.is_root()) {
        const int total_cells = total_values / ndim;
        global_result.resize(total_cells, ndim);
        for (int i = 0; i < total_cells; ++i) {
            for (int d = 0; d < ndim; ++d) {
                global_result(i, d) = recv_buf[i * ndim + d];
            }
        }
    }
#else
    global_result = local_field.values;
#endif
    return global_result;
}

} // namespace twofluid
