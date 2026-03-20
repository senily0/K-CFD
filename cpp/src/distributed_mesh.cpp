#include "twofluid/distributed_mesh.hpp"
#include <algorithm>
#include <set>
#include <iostream>

namespace twofluid {

// ---------------------------------------------------------------------------
// Ghost exchange for scalar field
// ---------------------------------------------------------------------------

void DistributedMesh::exchange_scalar(Eigen::VectorXd& values) const {
#ifdef USE_MPI
    for (const auto& layer : ghost_layers) {
        const int n_send = static_cast<int>(layer.send_cells.size());
        const int n_recv = static_cast<int>(layer.recv_cells.size());

        // Pack owned cell values to send
        std::vector<double> send_buf(n_send);
        for (int i = 0; i < n_send; ++i) {
            send_buf[i] = values[layer.send_cells[i]];
        }

        std::vector<double> recv_buf(n_recv);

        // Use MPI_Sendrecv to avoid deadlock
        MPI_Sendrecv(send_buf.data(), n_send, MPI_DOUBLE,
                     layer.remote_rank, 600,
                     recv_buf.data(), n_recv, MPI_DOUBLE,
                     layer.remote_rank, 600,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Unpack into ghost cell positions
        for (int i = 0; i < n_recv; ++i) {
            values[layer.recv_cells[i]] = recv_buf[i];
        }
    }
#else
    (void)values;
#endif
}

// ---------------------------------------------------------------------------
// Ghost exchange for vector field
// ---------------------------------------------------------------------------

void DistributedMesh::exchange_vector(Eigen::MatrixXd& values) const {
#ifdef USE_MPI
    const int ndim = static_cast<int>(values.cols());

    for (const auto& layer : ghost_layers) {
        const int n_send = static_cast<int>(layer.send_cells.size());
        const int n_recv = static_cast<int>(layer.recv_cells.size());

        // Pack
        std::vector<double> send_buf(n_send * ndim);
        for (int i = 0; i < n_send; ++i) {
            for (int d = 0; d < ndim; ++d) {
                send_buf[i * ndim + d] = values(layer.send_cells[i], d);
            }
        }

        std::vector<double> recv_buf(n_recv * ndim);

        MPI_Sendrecv(send_buf.data(), n_send * ndim, MPI_DOUBLE,
                     layer.remote_rank, 601,
                     recv_buf.data(), n_recv * ndim, MPI_DOUBLE,
                     layer.remote_rank, 601,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Unpack
        for (int i = 0; i < n_recv; ++i) {
            for (int d = 0; d < ndim; ++d) {
                values(layer.recv_cells[i], d) = recv_buf[i * ndim + d];
            }
        }
    }
#else
    (void)values;
#endif
}

// ---------------------------------------------------------------------------
// Global dot product: sum over OWNED cells only, then MPI_Allreduce
// ---------------------------------------------------------------------------

double DistributedMesh::global_dot(const Eigen::VectorXd& a,
                                    const Eigen::VectorXd& b) const {
    double local_dot = 0.0;
    for (int i = 0; i < n_owned; ++i) {
        local_dot += a[i] * b[i];
    }
    return global_sum(local_dot);
}

double DistributedMesh::global_norm(const Eigen::VectorXd& v) const {
    return std::sqrt(global_dot(v, v));
}

double DistributedMesh::global_sum(double val) const {
#ifdef USE_MPI
    double result = 0.0;
    MPI_Allreduce(&val, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return result;
#else
    return val;
#endif
}

double DistributedMesh::global_max(double val) const {
#ifdef USE_MPI
    double result = 0.0;
    MPI_Allreduce(&val, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
#else
    return val;
#endif
}

// ---------------------------------------------------------------------------
// Build DistributedMesh from global mesh + partition IDs
// ---------------------------------------------------------------------------

DistributedMesh build_distributed_mesh(
    const FVMesh& global_mesh,
    const std::vector<int>& part_ids,
    int my_rank,
    int n_ranks)
{
    DistributedMesh dm;
    FVMesh& lm = dm.local_mesh;
    lm = FVMesh(global_mesh.ndim);

    // ---------------------------------------------------------------
    // Step 1: Identify owned cells and ghost cells
    // ---------------------------------------------------------------

    // Owned cells: cells assigned to my_rank
    std::vector<int> owned_global;
    for (int ci = 0; ci < global_mesh.n_cells; ++ci) {
        if (part_ids[ci] == my_rank) {
            owned_global.push_back(ci);
        }
    }

    // Ghost cells: non-owned cells that share a face with an owned cell.
    // We need to discover these by scanning all global faces.
    std::set<int> ghost_global_set;
    for (int fi = 0; fi < global_mesh.n_faces; ++fi) {
        const Face& gf = global_mesh.faces[fi];
        if (gf.neighbour < 0) continue;  // boundary face

        bool owner_mine = (part_ids[gf.owner] == my_rank);
        bool neigh_mine = (part_ids[gf.neighbour] == my_rank);

        if (owner_mine && !neigh_mine) {
            ghost_global_set.insert(gf.neighbour);
        } else if (!owner_mine && neigh_mine) {
            ghost_global_set.insert(gf.owner);
        }
    }

    std::vector<int> ghost_global(ghost_global_set.begin(), ghost_global_set.end());
    std::sort(ghost_global.begin(), ghost_global.end());

    dm.n_owned = static_cast<int>(owned_global.size());
    dm.n_ghost = static_cast<int>(ghost_global.size());
    int n_total = dm.n_owned + dm.n_ghost;

    // ---------------------------------------------------------------
    // Step 2: Build global-to-local mapping
    // ---------------------------------------------------------------

    dm.local_to_global.resize(n_total);
    for (int i = 0; i < dm.n_owned; ++i) {
        dm.global_to_local[owned_global[i]] = i;
        dm.local_to_global[i] = owned_global[i];
    }
    for (int i = 0; i < dm.n_ghost; ++i) {
        int li = dm.n_owned + i;
        dm.global_to_local[ghost_global[i]] = li;
        dm.local_to_global[li] = ghost_global[i];
    }

    // ---------------------------------------------------------------
    // Step 3: Build local cells (owned + ghost)
    // ---------------------------------------------------------------

    // Collect used nodes
    std::set<int> used_nodes_set;
    for (int gi : owned_global) {
        for (int nid : global_mesh.cells[gi].nodes) used_nodes_set.insert(nid);
    }
    for (int gi : ghost_global) {
        for (int nid : global_mesh.cells[gi].nodes) used_nodes_set.insert(nid);
    }

    // Remap nodes
    std::unordered_map<int, int> node_map;
    int new_nid = 0;
    lm.nodes.resize(static_cast<int>(used_nodes_set.size()), 3);
    for (int old_nid : used_nodes_set) {
        node_map[old_nid] = new_nid;
        lm.nodes.row(new_nid) = global_mesh.nodes.row(old_nid);
        new_nid++;
    }

    // Create cells
    lm.n_cells = n_total;
    lm.cells.resize(n_total);

    auto setup_cell = [&](int local_idx, int global_idx) {
        auto& cell = lm.cells[local_idx];
        cell.center = global_mesh.cells[global_idx].center;
        cell.volume = global_mesh.cells[global_idx].volume;
        for (int nid : global_mesh.cells[global_idx].nodes) {
            auto it = node_map.find(nid);
            if (it != node_map.end()) cell.nodes.push_back(it->second);
        }
    };

    for (int i = 0; i < dm.n_owned; ++i) {
        setup_cell(i, owned_global[i]);
    }
    for (int i = 0; i < dm.n_ghost; ++i) {
        setup_cell(dm.n_owned + i, ghost_global[i]);
    }

    // ---------------------------------------------------------------
    // Step 4: Build local faces
    // ---------------------------------------------------------------
    // Scan all global faces. Three cases:
    //   a) Both cells owned by me -> internal face
    //   b) One cell owned, one ghost -> internal face (with ghost)
    //   c) Owner owned, neighbour == -1 (boundary) -> boundary face
    //   d) Neither cell owned by me -> skip

    for (int fi = 0; fi < global_mesh.n_faces; ++fi) {
        const Face& gf = global_mesh.faces[fi];

        auto it_o = dm.global_to_local.find(gf.owner);
        bool have_owner = (it_o != dm.global_to_local.end());

        if (gf.neighbour >= 0) {
            // Internal global face
            auto it_n = dm.global_to_local.find(gf.neighbour);
            bool have_neigh = (it_n != dm.global_to_local.end());

            if (have_owner && have_neigh) {
                // Both cells are in our local mesh (either owned or ghost)
                int lo = it_o->second;
                int ln = it_n->second;

                // Ensure owner < neighbour in local indexing for consistency
                // (OpenFOAM convention: owner < neighbour for internal faces)
                Vec3 normal = gf.normal;
                int face_owner = lo;
                int face_neigh = ln;
                if (lo > ln) {
                    face_owner = ln;
                    face_neigh = lo;
                    normal = Vec3(-gf.normal[0], -gf.normal[1], -gf.normal[2]);
                }

                int fid = static_cast<int>(lm.faces.size());
                Face lf;
                lf.owner = face_owner;
                lf.neighbour = face_neigh;
                lf.area = gf.area;
                lf.normal = normal;
                lf.center = gf.center;
                lf.boundary_tag = "";  // internal face
                for (int nid : gf.nodes) {
                    auto nit = node_map.find(nid);
                    if (nit != node_map.end()) lf.nodes.push_back(nit->second);
                }
                lm.faces.push_back(lf);
                lm.cells[face_owner].faces.push_back(fid);
                lm.cells[face_neigh].faces.push_back(fid);
            }
            // If only one is present but the other is not in global_to_local,
            // this shouldn't happen because we added all ghost cells.
        } else {
            // Boundary face in global mesh
            if (have_owner && part_ids[gf.owner] == my_rank) {
                // This boundary face belongs to an owned cell
                int lo = it_o->second;

                int fid = static_cast<int>(lm.faces.size());
                Face lf;
                lf.owner = lo;
                lf.neighbour = -1;  // boundary
                lf.area = gf.area;
                lf.normal = gf.normal;
                lf.center = gf.center;
                lf.boundary_tag = gf.boundary_tag;
                for (int nid : gf.nodes) {
                    auto nit = node_map.find(nid);
                    if (nit != node_map.end()) lf.nodes.push_back(nit->second);
                }
                lm.faces.push_back(lf);
                lm.cells[lo].faces.push_back(fid);
            }
        }
    }

    // ---------------------------------------------------------------
    // Step 5: Finalize mesh counts and boundary patches
    // ---------------------------------------------------------------

    lm.n_faces = static_cast<int>(lm.faces.size());
    lm.n_internal_faces = 0;
    lm.n_boundary_faces = 0;
    for (auto& f : lm.faces) {
        if (f.neighbour >= 0) lm.n_internal_faces++;
        else lm.n_boundary_faces++;
    }

    // Build boundary patches from face tags
    for (int fi = 0; fi < lm.n_faces; ++fi) {
        auto& f = lm.faces[fi];
        if (f.neighbour < 0 && !f.boundary_tag.empty()) {
            lm.boundary_patches[f.boundary_tag].push_back(fi);
        }
    }
    lm.build_boundary_face_cache();

    // ---------------------------------------------------------------
    // Step 6: Build ghost exchange layers
    // ---------------------------------------------------------------
    // Group ghost cells by their owning rank.
    // For each remote rank, we need to:
    //   - send: our owned cells that are ghost on the remote rank
    //   - recv: ghost cells on our mesh that are owned by the remote rank

    // First, build the recv side: for each ghost cell, which rank owns it?
    std::map<int, std::vector<int>> ghost_by_rank;  // rank -> list of global cell IDs
    std::map<int, std::vector<int>> ghost_local_by_rank;  // rank -> list of local cell indices
    for (int i = 0; i < dm.n_ghost; ++i) {
        int gi = ghost_global[i];
        int owner_rank = part_ids[gi];
        ghost_by_rank[owner_rank].push_back(gi);
        ghost_local_by_rank[owner_rank].push_back(dm.n_owned + i);
    }

#ifdef USE_MPI
    // For each neighbor rank, exchange the list of global cells needed.
    // This tells each rank which of its owned cells to pack and send.
    for (auto& [remote_rank, needed_globals] : ghost_by_rank) {
        GhostExchangeInfo info;
        info.remote_rank = remote_rank;
        info.recv_cells = ghost_local_by_rank[remote_rank];

        int n_i_need = static_cast<int>(needed_globals.size());
        int n_they_need = 0;

        MPI_Sendrecv(&n_i_need, 1, MPI_INT, remote_rank, 700,
                     &n_they_need, 1, MPI_INT, remote_rank, 700,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Exchange global cell ID lists
        std::vector<int> they_need(n_they_need);
        MPI_Sendrecv(needed_globals.data(), n_i_need, MPI_INT, remote_rank, 701,
                     they_need.data(), n_they_need, MPI_INT, remote_rank, 701,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Build send_cells: map global IDs the remote needs to local indices
        info.send_cells.resize(n_they_need);
        for (int i = 0; i < n_they_need; ++i) {
            auto it = dm.global_to_local.find(they_need[i]);
            if (it != dm.global_to_local.end()) {
                info.send_cells[i] = it->second;
            } else {
                info.send_cells[i] = 0;  // should not happen
            }
        }

        dm.ghost_layers.push_back(std::move(info));
    }
#else
    (void)n_ranks;
#endif

    return dm;
}

} // namespace twofluid
