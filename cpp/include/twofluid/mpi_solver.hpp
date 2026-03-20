#pragma once
#include <memory>
#include "twofluid/mpi_comm.hpp"
#include "twofluid/partitioning.hpp"
#include "twofluid/simple_solver.hpp"

namespace twofluid {

/// Distributed SIMPLE solver using MPI.
/// Each rank owns a local submesh and ghost cells.
class MPISIMPLESolver {
public:
    MPISIMPLESolver(MPIComm& comm, FVMesh& global_mesh, double rho, double mu);

    SolveResult solve_steady();

    // Access local solver
    SIMPLESolver& local_solver() { return *local_solver_; }

    // Gather solution to root rank
    Eigen::VectorXd gather_scalar(const ScalarField& local_field);
    Eigen::MatrixXd gather_vector(const VectorField& local_field);

private:
    MPIComm& comm_;
    LocalMesh local_mesh_;
    std::vector<GhostLayer> ghost_layers_;
    std::unique_ptr<SIMPLESolver> local_solver_;

    void exchange_ghosts(ScalarField& field);
    void exchange_ghosts(VectorField& field);
    double global_residual(double local_res);
};

} // namespace twofluid
