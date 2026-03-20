#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "twofluid/mesh.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace twofluid {

/// Ghost exchange info for one neighbor rank.
struct GhostExchangeInfo {
    int remote_rank;
    std::vector<int> send_cells;   // local owned cell indices to send
    std::vector<int> recv_cells;   // local ghost cell indices to receive into
};

/// Distributed mesh: local mesh with ghost cells appended after owned cells.
///
/// Ghost cells are real cells in local_mesh (with center, volume, faces).
/// Internal faces between owned and ghost cells are INTERNAL faces in
/// local_mesh (face.neighbour >= 0), so the SIMPLE solver treats them
/// as normal internal connections.
///
/// This is the OpenFOAM/ANSYS approach: the solver sees a single mesh
/// with n_owned + n_ghost cells, and ghost values are synchronized via
/// MPI exchange.
struct DistributedMesh {
    FVMesh local_mesh;       // includes ghost cells as real cells
    int n_owned = 0;         // cells [0, n_owned) are owned
    int n_ghost = 0;         // cells [n_owned, n_owned+n_ghost) are ghosts

    std::vector<GhostExchangeInfo> ghost_layers;

    // Mapping from global cell ID to local cell index
    std::unordered_map<int, int> global_to_local;
    // Mapping from local cell index to global cell ID
    std::vector<int> local_to_global;

    /// Exchange ghost values for a scalar field (VectorXd of size n_owned+n_ghost).
    void exchange_scalar(Eigen::VectorXd& values) const;

    /// Exchange ghost values for a vector field (MatrixXd of (n_owned+n_ghost, ndim)).
    void exchange_vector(Eigen::MatrixXd& values) const;

    /// Global dot product: local dot of owned portions + MPI_Allreduce SUM.
    double global_dot(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const;

    /// Global norm (L2): sqrt(global_dot(v, v)).
    double global_norm(const Eigen::VectorXd& v) const;

    /// Global sum of a scalar value.
    double global_sum(double val) const;

    /// Global max of a scalar value.
    double global_max(double val) const;
};

/// Build a DistributedMesh from the global mesh and partition IDs.
///
/// This function:
/// 1. Collects owned cells (part_ids[ci] == my_rank)
/// 2. For each global face between an owned cell and a non-owned cell,
///    adds the non-owned cell as a GHOST cell in the local mesh and
///    makes this face an INTERNAL face (owner=owned, neighbour=ghost)
/// 3. Ghost cells get center, volume from the global mesh
/// 4. Builds ghost_layers for MPI exchange
///
/// The key insight: the local_mesh has n_cells = n_owned + n_ghost,
/// and faces between owned and ghost cells are INTERNAL faces.
/// The SIMPLESolver sees them as normal internal connections.
DistributedMesh build_distributed_mesh(
    const FVMesh& global_mesh,
    const std::vector<int>& part_ids,
    int my_rank,
    int n_ranks);

} // namespace twofluid
