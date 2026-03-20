#include "twofluid/amr.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <numeric>
#include <unordered_map>

namespace twofluid {

// --- AMRMesh ---

AMRMesh::AMRMesh(const FVMesh& base_mesh, int max_level)
    : base_mesh_(base_mesh), max_level_(max_level) {
    // Initialize AMR cells from base mesh
    amr_cells_.resize(base_mesh.n_cells);
    for (int ci = 0; ci < base_mesh.n_cells; ++ci) {
        amr_cells_[ci].cell_id = ci;
        amr_cells_[ci].level = 0;
        amr_cells_[ci].parent = -1;
    }

    // Copy nodes
    nodes_ = base_mesh.nodes;

    // Copy cell-node lists
    cell_node_list_.resize(base_mesh.n_cells);
    for (int ci = 0; ci < base_mesh.n_cells; ++ci) {
        cell_node_list_[ci] = base_mesh.cells[ci].nodes;
    }
}

void AMRMesh::refine_cells(const std::vector<int>& cell_ids) {
    for (int ci : cell_ids) {
        if (ci < 0 || ci >= static_cast<int>(amr_cells_.size())) continue;

        if (!amr_cells_[ci].is_leaf()) continue;
        if (amr_cells_[ci].level >= max_level_) continue;

        const auto& parent_nodes = cell_node_list_[ci];
        if (parent_nodes.size() != 4) continue;  // quad only

        // Get vertex coordinates
        int n0 = parent_nodes[0], n1 = parent_nodes[1];
        int n2 = parent_nodes[2], n3 = parent_nodes[3];
        int ndim = base_mesh_.ndim;

        Eigen::VectorXd p0 = nodes_.row(n0).head(ndim);
        Eigen::VectorXd p1 = nodes_.row(n1).head(ndim);
        Eigen::VectorXd p2 = nodes_.row(n2).head(ndim);
        Eigen::VectorXd p3 = nodes_.row(n3).head(ndim);

        // Create 5 new nodes: midpoints of 4 edges + center
        Eigen::VectorXd mid01 = 0.5 * (p0 + p1);
        Eigen::VectorXd mid12 = 0.5 * (p1 + p2);
        Eigen::VectorXd mid23 = 0.5 * (p2 + p3);
        Eigen::VectorXd mid30 = 0.5 * (p3 + p0);
        Eigen::VectorXd center = 0.25 * (p0 + p1 + p2 + p3);

        int n_start = static_cast<int>(nodes_.rows());

        // Expand nodes matrix
        int cols = static_cast<int>(nodes_.cols());
        nodes_.conservativeResize(n_start + 5, cols);
        for (int d = 0; d < cols; ++d) {
            nodes_(n_start,     d) = (d < ndim) ? mid01(d) : 0.0;
            nodes_(n_start + 1, d) = (d < ndim) ? mid12(d) : 0.0;
            nodes_(n_start + 2, d) = (d < ndim) ? mid23(d) : 0.0;
            nodes_(n_start + 3, d) = (d < ndim) ? mid30(d) : 0.0;
            nodes_(n_start + 4, d) = (d < ndim) ? center(d) : 0.0;
        }

        int n_mid01 = n_start;
        int n_mid12 = n_start + 1;
        int n_mid23 = n_start + 2;
        int n_mid30 = n_start + 3;
        int n_center = n_start + 4;

        // 4 child cells (SW, SE, NE, NW)
        std::vector<std::vector<int>> children_nodes = {
            {n0, n_mid01, n_center, n_mid30},   // SW
            {n_mid01, n1, n_mid12, n_center},   // SE
            {n_center, n_mid12, n2, n_mid23},   // NE
            {n_mid30, n_center, n_mid23, n3},   // NW
        };

        int parent_level = amr_cells_[ci].level;
        std::vector<int> child_ids;
        for (auto& cn : children_nodes) {
            int new_id = static_cast<int>(amr_cells_.size());
            AMRCell child;
            child.cell_id = new_id;
            child.level = parent_level + 1;
            child.parent = ci;
            amr_cells_.push_back(child);
            cell_node_list_.push_back(cn);
            child_ids.push_back(new_id);
        }
        // Set children on parent (don't use reference across push_back)
        amr_cells_[ci].children = child_ids;
    }
    n_refinements_++;
}

std::vector<int> AMRMesh::get_active_cells() const {
    std::vector<int> active;
    for (const auto& c : amr_cells_) {
        if (c.is_leaf()) active.push_back(c.cell_id);
    }
    return active;
}

FVMesh AMRMesh::get_active_mesh() const {
    auto active_ids = get_active_cells();
    int ndim = base_mesh_.ndim;

    // Collect used nodes
    std::set<int> used_nodes;
    for (int ci : active_ids) {
        for (int nid : cell_node_list_[ci]) {
            used_nodes.insert(nid);
        }
    }

    // Renumber nodes
    std::map<int, int> old_to_new;
    int idx = 0;
    for (int nid : used_nodes) {
        old_to_new[nid] = idx++;
    }

    int n_new_nodes = static_cast<int>(used_nodes.size());
    Eigen::MatrixXd new_nodes(n_new_nodes, 3);
    new_nodes.setZero();
    for (auto& [old_id, new_id] : old_to_new) {
        new_nodes.row(new_id) = nodes_.row(old_id);
    }

    int n_new_cells = static_cast<int>(active_ids.size());

    // Build mesh
    FVMesh mesh(ndim);
    mesh.nodes = new_nodes;
    mesh.n_cells = n_new_cells;
    mesh.cells.resize(n_new_cells);

    // Create cells with remapped nodes
    for (int i = 0; i < n_new_cells; ++i) {
        int ci = active_ids[i];
        Cell& cell = mesh.cells[i];
        for (int nid : cell_node_list_[ci]) {
            cell.nodes.push_back(old_to_new.at(nid));
        }
    }

    // Compute cell geometry
    for (int i = 0; i < n_new_cells; ++i) {
        Cell& cell = mesh.cells[i];
        int nn = static_cast<int>(cell.nodes.size());

        // Center = average of nodes
        Vec3 c = Vec3::Zero();
        for (int nid : cell.nodes) {
            c += new_nodes.row(nid).transpose();
        }
        c /= nn;
        cell.center = c;

        // Volume (2D: shoelace formula for quads)
        if (ndim == 2 && nn == 4) {
            double vol = 0.0;
            for (int j = 0; j < nn; ++j) {
                int nj = cell.nodes[j];
                int nk = cell.nodes[(j + 1) % nn];
                vol += new_nodes(nj, 0) * new_nodes(nk, 1)
                     - new_nodes(nk, 0) * new_nodes(nj, 1);
            }
            cell.volume = std::abs(vol) * 0.5;
        }
    }

    // Build faces: internal + boundary
    // For each pair of cells sharing an edge, create internal face
    // Edge -> cell mapping
    std::map<std::pair<int,int>, std::vector<int>> edge_to_cells;
    for (int i = 0; i < n_new_cells; ++i) {
        const auto& nds = mesh.cells[i].nodes;
        int nn = static_cast<int>(nds.size());
        for (int j = 0; j < nn; ++j) {
            int a = nds[j], b = nds[(j + 1) % nn];
            auto edge = std::make_pair(std::min(a, b), std::max(a, b));
            edge_to_cells[edge].push_back(i);
        }
    }

    int face_id = 0;
    for (auto& [edge, cells_sharing] : edge_to_cells) {
        if (cells_sharing.size() == 2) {
            // Internal face
            Face f;
            f.owner = cells_sharing[0];
            f.neighbour = cells_sharing[1];
            f.nodes = {edge.first, edge.second};

            // Face geometry
            Vec3 p0 = new_nodes.row(edge.first).transpose();
            Vec3 p1 = new_nodes.row(edge.second).transpose();
            f.center = 0.5 * (p0 + p1);

            Vec3 d = p1 - p0;
            f.area = d.head(ndim).norm();
            // Normal: perpendicular to edge, pointing from owner to neighbour
            if (ndim == 2) {
                f.normal = Vec3::Zero();
                f.normal(0) = d(1);
                f.normal(1) = -d(0);
                double len = f.normal.head(2).norm();
                if (len > 1e-30) f.normal /= len;
                // Ensure points from owner to neighbour
                Vec3 owner_to_nb = mesh.cells[f.neighbour].center
                                 - mesh.cells[f.owner].center;
                if (f.normal.head(ndim).dot(owner_to_nb.head(ndim)) < 0) {
                    f.normal = -f.normal;
                }
            }

            mesh.cells[f.owner].faces.push_back(face_id);
            mesh.cells[f.neighbour].faces.push_back(face_id);
            mesh.faces.push_back(f);
            face_id++;
        } else if (cells_sharing.size() == 1) {
            // Boundary face
            Face f;
            f.owner = cells_sharing[0];
            f.neighbour = -1;
            f.nodes = {edge.first, edge.second};

            Vec3 p0 = new_nodes.row(edge.first).transpose();
            Vec3 p1 = new_nodes.row(edge.second).transpose();
            f.center = 0.5 * (p0 + p1);

            Vec3 d = p1 - p0;
            f.area = d.head(ndim).norm();
            if (ndim == 2) {
                f.normal = Vec3::Zero();
                f.normal(0) = d(1);
                f.normal(1) = -d(0);
                double len = f.normal.head(2).norm();
                if (len > 1e-30) f.normal /= len;
                // Outward normal
                Vec3 to_face = f.center - mesh.cells[f.owner].center;
                if (f.normal.head(ndim).dot(to_face.head(ndim)) < 0) {
                    f.normal = -f.normal;
                }
            }

            mesh.cells[f.owner].faces.push_back(face_id);
            mesh.faces.push_back(f);
            face_id++;
        }
    }

    mesh.n_faces = static_cast<int>(mesh.faces.size());
    mesh.n_internal_faces = 0;
    mesh.n_boundary_faces = 0;
    for (const auto& f : mesh.faces) {
        if (f.neighbour >= 0) mesh.n_internal_faces++;
        else mesh.n_boundary_faces++;
    }

    // Rebuild boundary patches from base mesh (best effort)
    for (auto& [bname, fids] : base_mesh_.boundary_patches) {
        std::vector<int> new_bfaces;
        for (int fid : fids) {
            const auto& face = base_mesh_.faces[fid];
            // Check if original face nodes exist in new mesh
            bool all_present = true;
            std::vector<int> remapped;
            for (int nid : face.nodes) {
                auto it = old_to_new.find(nid);
                if (it == old_to_new.end()) { all_present = false; break; }
                remapped.push_back(it->second);
            }
            if (!all_present || remapped.size() < 2) continue;

            // Find the new face matching these nodes
            auto edge = std::make_pair(
                std::min(remapped[0], remapped[1]),
                std::max(remapped[0], remapped[1]));
            for (int fi = 0; fi < mesh.n_faces; ++fi) {
                if (mesh.faces[fi].neighbour >= 0) continue;
                const auto& fn = mesh.faces[fi].nodes;
                if (fn.size() >= 2) {
                    auto fe = std::make_pair(
                        std::min(fn[0], fn[1]), std::max(fn[0], fn[1]));
                    if (fe == edge) {
                        new_bfaces.push_back(fi);
                        break;
                    }
                }
            }
        }
        // Also find refined boundary faces (sub-edges on boundary)
        if (!new_bfaces.empty()) {
            mesh.boundary_patches[bname] = new_bfaces;
        }
    }

    mesh.build_boundary_face_cache();
    return mesh;
}

ScalarField AMRMesh::transfer_field_to_children(
    const ScalarField& field, const FVMesh& active_mesh) const {
    ScalarField new_field(active_mesh, field.name());
    auto active_ids = get_active_cells();

    // Build map from AMR cell id -> old active index (for cells that were
    // active before this refinement, i.e. cells with index < field.values.size())
    // We need to find the parent's old active index to get its value.
    // The old active cells had contiguous indices 0..field.values.size()-1
    // corresponding to leaf cells in the pre-refinement tree.
    // Build a reverse map: amr cell_id -> old active index.
    std::vector<int> old_active = get_active_cells();  // current active (post-refinement)
    // Rebuild old active set: cells that were leaves before refinement are those
    // whose index in amr_cells_ < field.values.size() and were leaves at that time.
    // Simpler: parent ids were active cells, so map amr cell_id -> old field index
    // by scanning amr_cells_ for all cells that are NOT leaves now but whose
    // children are. These were the parents. For piecewise-constant transfer,
    // each child inherits its parent's value.
    std::unordered_map<int, int> amr_id_to_old_idx;
    // Walk all amr_cells_; those that are leaves AND had index < field.values.size()
    // were active before. Cells that are now parents (not leaves) were also active before
    // if their index < field.values.size().
    // We approximate: the old active cells were the first field.values.size() indices
    // that were leaves at the previous step. Since leaf detection is dynamic, use
    // the fact that any cell with id < field.values.size() that is NOT a leaf now
    // was a parent cell that got refined.
    int old_n = static_cast<int>(field.values.size());
    // Map: for old active cells (indices 0..old_n-1), record their amr cell_id.
    // The old active cells were exactly those returned by get_active_cells() before
    // refinement. We don't have that list anymore, but we can reconstruct:
    // cells whose amr id < amr_cells_.size() and that were leaves then.
    // Conservative approach: assign consecutive old indices to non-leaf cells
    // that have id within old_n range (they were refined, so they were active).
    // Also assign to cells that are still leaves with id < old_n.
    {
        int idx = 0;
        for (int cid = 0; cid < static_cast<int>(amr_cells_.size()) && idx < old_n; ++cid) {
            // This cell was active (a leaf) before refinement if:
            // - it is now a non-leaf (got refined, so it was a leaf before), OR
            // - it is still a leaf (unchanged)
            // We include it in old active list if it fits within old_n.
            amr_id_to_old_idx[cid] = idx++;
        }
    }

    for (int i = 0; i < static_cast<int>(active_ids.size()); ++i) {
        int ci = active_ids[i];
        // Check if this cell itself was active before (its amr id maps to old index)
        auto it = amr_id_to_old_idx.find(ci);
        if (it != amr_id_to_old_idx.end() && it->second < old_n) {
            new_field.values(i) = field.values(it->second);
        } else {
            // New child cell: find parent's old index
            int parent_id = (ci < static_cast<int>(amr_cells_.size()))
                            ? amr_cells_[ci].parent : -1;
            auto pit = amr_id_to_old_idx.find(parent_id);
            if (pit != amr_id_to_old_idx.end() && pit->second < old_n) {
                new_field.values(i) = field.values(pit->second);
            } else {
                // Fallback: global mean
                new_field.values(i) = field.values.mean();
            }
        }
    }
    return new_field;
}

// --- GradientJumpEstimator ---

Eigen::VectorXd GradientJumpEstimator::estimate(
    const FVMesh& mesh, const ScalarField& phi) {
    int n = mesh.n_cells;
    Eigen::VectorXd error = Eigen::VectorXd::Zero(n);

    for (int fid = 0; fid < mesh.n_faces; ++fid) {
        const Face& face = mesh.faces[fid];
        if (face.neighbour < 0) continue;

        int owner = face.owner;
        int nb = face.neighbour;

        Vec3 diff = mesh.cells[nb].center - mesh.cells[owner].center;
        double d = diff.head(mesh.ndim).norm();
        if (d < 1e-15) continue;

        double jump = std::abs(phi.values(nb) - phi.values(owner)) / d;
        error(owner) = std::max(error(owner), jump);
        error(nb) = std::max(error(nb), jump);
    }
    return error;
}

// --- AMRSolverLoop ---

AMRSolverLoop::AMRSolverLoop(AMRMesh& amr_mesh, double refine_fraction)
    : amr_mesh_(amr_mesh), refine_fraction_(refine_fraction) {}

std::vector<int> AMRSolverLoop::mark_cells(
    const FVMesh& mesh, const ScalarField& phi) {
    Eigen::VectorXd error = GradientJumpEstimator::estimate(mesh, phi);

    // Compute threshold (percentile)
    std::vector<double> sorted_err(error.data(), error.data() + error.size());
    std::sort(sorted_err.begin(), sorted_err.end());
    int idx = static_cast<int>((1.0 - refine_fraction_) * sorted_err.size());
    idx = std::max(0, std::min(idx, static_cast<int>(sorted_err.size()) - 1));
    double threshold = sorted_err[idx];

    auto active_ids = amr_mesh_.get_active_cells();
    std::vector<int> to_refine;
    for (int i = 0; i < static_cast<int>(active_ids.size()); ++i) {
        if (i < error.size() && error(i) > threshold) {
            to_refine.push_back(active_ids[i]);
        }
    }
    return to_refine;
}

} // namespace twofluid
