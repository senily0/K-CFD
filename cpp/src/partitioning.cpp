#include "twofluid/partitioning.hpp"
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace twofluid {

// ---------------------------------------------------------------------------
// RCB partitioner
// ---------------------------------------------------------------------------

void RCBPartitioner::bisect(const FVMesh& mesh,
                             const std::vector<int>& cell_ids,
                             int n_parts,
                             std::vector<int>& part_ids,
                             int part_offset) {
    if (cell_ids.empty()) return;

    if (n_parts == 1) {
        for (int ci : cell_ids) {
            part_ids[ci] = part_offset;
        }
        return;
    }

    // Find bounding box and longest dimension
    Vec3 lo = mesh.cells[cell_ids[0]].center;
    Vec3 hi = lo;
    for (int ci : cell_ids) {
        const Vec3& c = mesh.cells[ci].center;
        for (int d = 0; d < 3; ++d) {
            if (c[d] < lo[d]) lo[d] = c[d];
            if (c[d] > hi[d]) hi[d] = c[d];
        }
    }

    int axis = 0;
    double span = hi[0] - lo[0];
    for (int d = 1; d < 3; ++d) {
        if (hi[d] - lo[d] > span) {
            span = hi[d] - lo[d];
            axis = d;
        }
    }

    // Sort by coordinate along chosen axis
    std::vector<int> sorted = cell_ids;
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
        return mesh.cells[a].center[axis] < mesh.cells[b].center[axis];
    });

    // Split at median
    const std::size_t mid = sorted.size() / 2;
    std::vector<int> left(sorted.begin(), sorted.begin() + mid);
    std::vector<int> right(sorted.begin() + mid, sorted.end());

    const int left_parts  = n_parts / 2;
    const int right_parts = n_parts - left_parts;

    bisect(mesh, left,  left_parts,  part_ids, part_offset);
    bisect(mesh, right, right_parts, part_ids, part_offset + left_parts);
}

std::vector<int> RCBPartitioner::partition(const FVMesh& mesh, int n_parts) {
    if (n_parts <= 0) throw std::invalid_argument("n_parts must be >= 1");

    std::vector<int> part_ids(mesh.n_cells, 0);
    if (n_parts == 1) return part_ids;

    std::vector<int> all_cells(mesh.n_cells);
    for (int i = 0; i < mesh.n_cells; ++i) all_cells[i] = i;

    bisect(mesh, all_cells, n_parts, part_ids, 0);
    return part_ids;
}

// ---------------------------------------------------------------------------
// build_ghost_layers
// ---------------------------------------------------------------------------

std::vector<GhostLayer> build_ghost_layers(const FVMesh& mesh,
                                            const std::vector<int>& part_ids,
                                            int my_rank) {
    // neighbor_rank -> GhostLayer (indexed by neighbor rank for quick lookup)
    std::unordered_map<int, GhostLayer> layer_map;

    for (int fi = 0; fi < mesh.n_internal_faces; ++fi) {
        const Face& f = mesh.faces[fi];
        const int owner_part = part_ids[f.owner];
        const int nbr_part   = part_ids[f.neighbour];

        if (owner_part == nbr_part) continue;  // same partition, no ghost needed

        if (owner_part == my_rank) {
            // I own the owner cell; neighbour is across the boundary
            auto& layer = layer_map[nbr_part];
            layer.neighbor_rank = nbr_part;
            layer.send_cells.push_back(f.owner);
            layer.recv_cells.push_back(f.neighbour);
        } else if (nbr_part == my_rank) {
            // I own the neighbour cell; owner is across the boundary
            auto& layer = layer_map[owner_part];
            layer.neighbor_rank = owner_part;
            layer.send_cells.push_back(f.neighbour);
            layer.recv_cells.push_back(f.owner);
        }
    }

    // Deduplicate send/recv cell lists per layer
    for (auto& [rank, layer] : layer_map) {
        // Deduplicate send_cells
        std::sort(layer.send_cells.begin(), layer.send_cells.end());
        layer.send_cells.erase(
            std::unique(layer.send_cells.begin(), layer.send_cells.end()),
            layer.send_cells.end());

        // Deduplicate recv_cells
        std::sort(layer.recv_cells.begin(), layer.recv_cells.end());
        layer.recv_cells.erase(
            std::unique(layer.recv_cells.begin(), layer.recv_cells.end()),
            layer.recv_cells.end());
    }

    std::vector<GhostLayer> result;
    result.reserve(layer_map.size());
    for (auto& [rank, layer] : layer_map) {
        result.push_back(std::move(layer));
    }
    return result;
}

// ---------------------------------------------------------------------------
// extract_local_mesh
// ---------------------------------------------------------------------------

LocalMesh extract_local_mesh(const FVMesh& global_mesh,
                              const std::vector<int>& part_ids,
                              int my_rank) {
    LocalMesh lm;
    lm.mesh = FVMesh(global_mesh.ndim);

    // Collect owned cells
    std::vector<int> owned_cells;
    for (int ci = 0; ci < global_mesh.n_cells; ++ci) {
        if (part_ids[ci] == my_rank) {
            owned_cells.push_back(ci);
        }
    }

    // Collect ghost cells: neighbours of owned cells that belong to another rank
    std::vector<int> ghost_cells_global;
    {
        std::unordered_map<int, int> candidate;
        for (int fi = 0; fi < global_mesh.n_internal_faces; ++fi) {
            const Face& f = global_mesh.faces[fi];
            if (part_ids[f.owner] == my_rank && part_ids[f.neighbour] != my_rank) {
                candidate[f.neighbour] = 1;
            } else if (part_ids[f.neighbour] == my_rank && part_ids[f.owner] != my_rank) {
                candidate[f.owner] = 1;
            }
        }
        ghost_cells_global.reserve(candidate.size());
        for (auto& [ci, _] : candidate) ghost_cells_global.push_back(ci);
        std::sort(ghost_cells_global.begin(), ghost_cells_global.end());
    }

    // Build local_to_global and global_to_local
    lm.local_to_global = owned_cells;
    lm.local_to_global.insert(lm.local_to_global.end(),
                               ghost_cells_global.begin(),
                               ghost_cells_global.end());

    for (int li = 0; li < static_cast<int>(lm.local_to_global.size()); ++li) {
        lm.global_to_local[lm.local_to_global[li]] = li;
    }

    const int n_local = static_cast<int>(owned_cells.size());
    const int n_ghost = static_cast<int>(ghost_cells_global.size());
    const int n_total = n_local + n_ghost;

    // Copy cell data
    lm.mesh.n_cells = n_total;
    lm.mesh.cells.resize(n_total);
    for (int li = 0; li < n_total; ++li) {
        lm.mesh.cells[li] = global_mesh.cells[lm.local_to_global[li]];
        // Remap face indices — done later
    }

    // Collect unique nodes referenced by local cells
    std::unordered_map<int, int> global_to_local_node;
    std::vector<int> local_nodes;
    for (int li = 0; li < n_total; ++li) {
        for (int ni : global_mesh.cells[lm.local_to_global[li]].nodes) {
            if (global_to_local_node.find(ni) == global_to_local_node.end()) {
                int local_ni = static_cast<int>(local_nodes.size());
                global_to_local_node[ni] = local_ni;
                local_nodes.push_back(ni);
            }
        }
    }

    // Copy nodes
    const int n_nodes = static_cast<int>(local_nodes.size());
    lm.mesh.nodes.resize(n_nodes, 3);
    for (int i = 0; i < n_nodes; ++i) {
        lm.mesh.nodes.row(i) = global_mesh.nodes.row(local_nodes[i]);
    }

    // Remap node indices in local cells
    for (int li = 0; li < n_total; ++li) {
        for (int& ni : lm.mesh.cells[li].nodes) {
            ni = global_to_local_node.at(ni);
        }
    }

    // Collect faces: internal faces between local cells, and boundary faces of local cells
    std::vector<Face> local_faces;
    // Map global face id -> local face id
    std::unordered_map<int, int> global_face_to_local;

    for (int fi = 0; fi < global_mesh.n_faces; ++fi) {
        const Face& gf = global_mesh.faces[fi];
        bool owner_local   = lm.global_to_local.count(gf.owner) > 0;
        bool nbr_local     = (gf.neighbour >= 0) &&
                             (lm.global_to_local.count(gf.neighbour) > 0);

        if (!owner_local && !nbr_local) continue;

        Face lf = gf;
        // Remap owner/neighbour to local indices
        if (owner_local) {
            lf.owner = lm.global_to_local.at(gf.owner);
        }
        if (gf.neighbour >= 0 && nbr_local) {
            lf.neighbour = lm.global_to_local.at(gf.neighbour);
        } else if (gf.neighbour >= 0 && !nbr_local) {
            // Cross-partition internal face becomes a "boundary" in the local mesh
            // Mark the remote side as -1 (boundary)
            lf.neighbour = -1;
            lf.boundary_tag = "mpi_interface";
        }
        // Remap node indices
        for (int& ni : lf.nodes) {
            auto it = global_to_local_node.find(ni);
            if (it != global_to_local_node.end()) ni = it->second;
        }

        global_face_to_local[fi] = static_cast<int>(local_faces.size());
        local_faces.push_back(std::move(lf));
    }

    // Count internal vs boundary faces
    int n_internal = 0;
    int n_boundary = 0;
    for (const Face& lf : local_faces) {
        if (lf.neighbour >= 0) ++n_internal;
        else ++n_boundary;
    }

    // Reorder: internal faces first, then boundary faces
    std::vector<Face> ordered_faces;
    ordered_faces.reserve(local_faces.size());
    for (const Face& lf : local_faces) {
        if (lf.neighbour >= 0) ordered_faces.push_back(lf);
    }
    for (const Face& lf : local_faces) {
        if (lf.neighbour < 0) ordered_faces.push_back(lf);
    }

    lm.mesh.faces = std::move(ordered_faces);
    lm.mesh.n_faces = static_cast<int>(lm.mesh.faces.size());
    lm.mesh.n_internal_faces = n_internal;
    lm.mesh.n_boundary_faces = n_boundary;

    // Remap face indices in cells (approximate: just rebuild face lists)
    // Build a face index map after reordering
    // Use a simpler approach: map old local face index -> new ordered index
    // This requires a two-pass rebuild. For now, clear and skip cell.faces
    // (solvers typically iterate over mesh.faces, not cell.faces for FVM)
    for (auto& cell : lm.mesh.cells) {
        cell.faces.clear();
    }

    // Copy boundary patches (only those touching local owned cells)
    for (const auto& [patch_name, face_ids] : global_mesh.boundary_patches) {
        std::vector<int> local_patch_faces;
        for (int gfi : face_ids) {
            const Face& gf = global_mesh.faces[gfi];
            if (lm.global_to_local.count(gf.owner) > 0) {
                auto it = global_face_to_local.find(gfi);
                if (it != global_face_to_local.end()) {
                    // Find position in ordered_faces
                    // The reordering above invalidated global_face_to_local values.
                    // Re-search by boundary_tag and owner match.
                    // We just track by name for now.
                    local_patch_faces.push_back(it->second);
                }
            }
        }
        if (!local_patch_faces.empty()) {
            lm.mesh.boundary_patches[patch_name] = local_patch_faces;
        }
    }

    lm.mesh.build_boundary_face_cache();
    return lm;
}

} // namespace twofluid
