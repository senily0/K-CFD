#include "twofluid/partitioning.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>

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

// ---------------------------------------------------------------------------
// GraphPartitioner — multilevel k-way partitioner (METIS-like, no external lib)
// ---------------------------------------------------------------------------

GraphPartitioner::Graph GraphPartitioner::build_dual_graph(const FVMesh& mesh) {
    const int nv = mesh.n_cells;
    // Count neighbours per cell for xadj
    std::vector<int> degree(nv, 0);
    for (int fi = 0; fi < mesh.n_internal_faces; ++fi) {
        const Face& f = mesh.faces[fi];
        ++degree[f.owner];
        ++degree[f.neighbour];
    }

    Graph g;
    g.n_vertices = nv;
    g.xadj.resize(nv + 1, 0);
    for (int i = 0; i < nv; ++i) g.xadj[i + 1] = g.xadj[i] + degree[i];
    g.adjncy.resize(g.xadj[nv]);
    g.adjwgt.resize(g.xadj[nv], 1);

    // Fill adjacency using fill pointer
    std::vector<int> fill(nv, 0);
    for (int fi = 0; fi < mesh.n_internal_faces; ++fi) {
        const Face& f = mesh.faces[fi];
        int o = f.owner, nb = f.neighbour;
        g.adjncy[g.xadj[o] + fill[o]++] = nb;
        g.adjncy[g.xadj[nb] + fill[nb]++] = o;
    }

    // Copy cell centres
    g.cx.resize(nv); g.cy.resize(nv); g.cz.resize(nv);
    for (int i = 0; i < nv; ++i) {
        g.cx[i] = mesh.cells[i].center[0];
        g.cy[i] = mesh.cells[i].center[1];
        g.cz[i] = mesh.cells[i].center[2];
    }
    return g;
}

// Coarsen by heavy-edge matching (greedy: first unmatched neighbour wins).
// coarse_to_fine[c] = list of fine vertices merged into coarse vertex c.
GraphPartitioner::Graph GraphPartitioner::coarsen(
        const Graph& g, std::vector<std::vector<int>>& coarse_to_fine) {

    const int nv = g.n_vertices;
    std::vector<int> match(nv, -1);  // fine -> coarse id
    int nc = 0;                       // number of coarse vertices

    // Greedy matching: iterate vertices in natural order
    for (int v = 0; v < nv; ++v) {
        if (match[v] != -1) continue;
        // Try to match with first unmatched neighbour
        int partner = -1;
        for (int j = g.xadj[v]; j < g.xadj[v + 1]; ++j) {
            int u = g.adjncy[j];
            if (match[u] == -1) { partner = u; break; }
        }
        if (partner == -1) {
            // No unmatched neighbour — singleton super-vertex
            match[v] = nc++;
        } else {
            match[v] = match[partner] = nc++;
        }
    }

    coarse_to_fine.assign(nc, {});
    for (int v = 0; v < nv; ++v) coarse_to_fine[match[v]].push_back(v);

    // Build coarsened graph in CSR
    // For each coarse vertex pair, sum edge weights of fine edges between them
    // Use a temporary adjacency map per coarse vertex
    std::vector<std::unordered_map<int, int>> cadj(nc);
    for (int v = 0; v < nv; ++v) {
        int cv = match[v];
        for (int j = g.xadj[v]; j < g.xadj[v + 1]; ++j) {
            int u = g.adjncy[j];
            int cu = match[u];
            if (cv != cu) cadj[cv][cu] += g.adjwgt[j];
        }
    }

    Graph cg;
    cg.n_vertices = nc;
    cg.xadj.resize(nc + 1, 0);
    for (int cv = 0; cv < nc; ++cv) cg.xadj[cv + 1] = cg.xadj[cv] + static_cast<int>(cadj[cv].size());
    cg.adjncy.resize(cg.xadj[nc]);
    cg.adjwgt.resize(cg.xadj[nc]);
    cg.cx.resize(nc); cg.cy.resize(nc); cg.cz.resize(nc);

    for (int cv = 0; cv < nc; ++cv) {
        // Average centre of constituent fine vertices
        double sx = 0, sy = 0, sz = 0;
        for (int fv : coarse_to_fine[cv]) { sx += g.cx[fv]; sy += g.cy[fv]; sz += g.cz[fv]; }
        double inv = 1.0 / static_cast<int>(coarse_to_fine[cv].size());
        cg.cx[cv] = sx * inv; cg.cy[cv] = sy * inv; cg.cz[cv] = sz * inv;

        int pos = cg.xadj[cv];
        for (auto& [cu, wt] : cadj[cv]) {
            cg.adjncy[pos] = cu;
            cg.adjwgt[pos] = wt;
            ++pos;
        }
    }
    return cg;
}

// BFS-based bisection: find two antipodal vertices as seeds, then grow two
// regions greedily to balance partition sizes.
std::vector<int> GraphPartitioner::initial_bisect(const Graph& g) {
    const int nv = g.n_vertices;
    if (nv == 0) return {};

    // Find seed 0: vertex with extreme coordinate in the widest axis
    double xspan = *std::max_element(g.cx.begin(), g.cx.end()) - *std::min_element(g.cx.begin(), g.cx.end());
    double yspan = *std::max_element(g.cy.begin(), g.cy.end()) - *std::min_element(g.cy.begin(), g.cy.end());
    double zspan = *std::max_element(g.cz.begin(), g.cz.end()) - *std::min_element(g.cz.begin(), g.cz.end());

    const std::vector<double>* coord = &g.cx;
    if (yspan > xspan && yspan >= zspan) coord = &g.cy;
    else if (zspan > xspan && zspan > yspan) coord = &g.cz;

    int seed0 = static_cast<int>(std::min_element(coord->begin(), coord->end()) - coord->begin());
    int seed1 = static_cast<int>(std::max_element(coord->begin(), coord->end()) - coord->begin());

    // BFS from both seeds simultaneously; each vertex gets label 0 or 1
    std::vector<int> parts(nv, -1);
    std::queue<int> q;
    parts[seed0] = 0; q.push(seed0);
    parts[seed1] = 1; q.push(seed1);

    const int target = nv / 2;
    int cnt0 = 1, cnt1 = 1;

    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int j = g.xadj[v]; j < g.xadj[v + 1]; ++j) {
            int u = g.adjncy[j];
            if (parts[u] == -1) {
                // Assign to the smaller partition, but cap at target
                int p;
                if (cnt0 < target) p = 0;
                else if (cnt1 < nv - target) p = 1;
                else p = (parts[v] == 0) ? 0 : 1;  // follow parent
                parts[u] = p;
                if (p == 0) ++cnt0; else ++cnt1;
                q.push(u);
            }
        }
    }

    // Any remaining unvisited (disconnected) vertices go to part 0
    for (int v = 0; v < nv; ++v) if (parts[v] == -1) parts[v] = 0;

    return parts;
}

// Fiduccia-Mattheyses single pass: iterate boundary vertices, swap those
// with positive gain. Repeat until no improvement.
void GraphPartitioner::kl_refine(const Graph& g, std::vector<int>& parts) {
    const int nv = g.n_vertices;
    bool improved = true;
    const int max_passes = 10;

    for (int pass = 0; pass < max_passes && improved; ++pass) {
        improved = false;

        for (int v = 0; v < nv; ++v) {
            int pv = parts[v];
            // Check if v is on boundary (has a neighbour in the other part)
            bool boundary = false;
            int ext = 0, intr = 0;
            for (int j = g.xadj[v]; j < g.xadj[v + 1]; ++j) {
                int u = g.adjncy[j];
                int w = g.adjwgt[j];
                if (parts[u] != pv) { ext += w; boundary = true; }
                else                { intr += w; }
            }
            if (!boundary) continue;

            // gain = (edges to other side) - (edges to same side)
            int gain = ext - intr;
            if (gain > 0) {
                // Count sizes
                int cnt0 = 0, cnt1 = 0;
                for (int p : parts) { if (p == 0) ++cnt0; else ++cnt1; }
                // Only swap if balance stays within 20% of ideal
                int target0 = nv / 2;
                int new_cnt_pv   = (pv == 0) ? cnt0 - 1 : cnt1 - 1;
                int new_cnt_npv  = (pv == 0) ? cnt1 + 1 : cnt0 + 1;
                int imb_before = std::abs(cnt0 - cnt1);
                int imb_after  = std::abs(new_cnt_pv - new_cnt_npv);
                (void)target0;
                // Accept if gain positive and balance doesn't worsen beyond threshold
                if (imb_after <= std::max(imb_before, nv / 5 + 1)) {
                    parts[v] = 1 - pv;
                    improved = true;
                }
            }
        }
    }
}

std::vector<int> GraphPartitioner::multilevel_bisect(const Graph& g) {
    if (g.n_vertices <= 1) {
        return std::vector<int>(g.n_vertices, 0);
    }
    if (g.n_vertices <= 4) {
        return initial_bisect(g);
    }

    // Coarsen until small enough or no reduction possible
    constexpr int COARSEN_LIMIT = 100;
    std::vector<Graph> hierarchy;
    std::vector<std::vector<std::vector<int>>> c2f_stack;  // [level][coarse_v] = list of fine v

    hierarchy.push_back(g);
    while (hierarchy.back().n_vertices > COARSEN_LIMIT) {
        std::vector<std::vector<int>> c2f;
        Graph cg = coarsen(hierarchy.back(), c2f);
        if (cg.n_vertices >= hierarchy.back().n_vertices) break;  // no progress
        c2f_stack.push_back(std::move(c2f));
        hierarchy.push_back(std::move(cg));
    }

    // Bisect coarsest graph
    std::vector<int> parts = initial_bisect(hierarchy.back());
    kl_refine(hierarchy.back(), parts);

    // Uncoarsen: project partition back through hierarchy levels
    for (int lvl = static_cast<int>(c2f_stack.size()) - 1; lvl >= 0; --lvl) {
        const auto& c2f = c2f_stack[lvl];
        const Graph& fine_g = hierarchy[lvl];
        const int nfine = fine_g.n_vertices;
        std::vector<int> fine_parts(nfine);
        for (int cv = 0; cv < static_cast<int>(c2f.size()); ++cv) {
            for (int fv : c2f[cv]) fine_parts[fv] = parts[cv];
        }
        kl_refine(fine_g, fine_parts);
        parts = std::move(fine_parts);
    }

    return parts;
}

GraphPartitioner::Graph GraphPartitioner::subgraph(
        const Graph& g, const std::vector<int>& verts,
        std::vector<int>& local_to_global) {
    local_to_global = verts;
    const int nv = static_cast<int>(verts.size());

    // Build global->local map
    std::unordered_map<int, int> g2l;
    g2l.reserve(nv * 2);
    for (int i = 0; i < nv; ++i) g2l[verts[i]] = i;

    Graph sg;
    sg.n_vertices = nv;
    sg.xadj.resize(nv + 1, 0);
    sg.cx.resize(nv); sg.cy.resize(nv); sg.cz.resize(nv);

    for (int li = 0; li < nv; ++li) {
        int gi = verts[li];
        sg.cx[li] = g.cx[gi]; sg.cy[li] = g.cy[gi]; sg.cz[li] = g.cz[gi];
        int deg = 0;
        for (int j = g.xadj[gi]; j < g.xadj[gi + 1]; ++j) {
            if (g2l.count(g.adjncy[j])) ++deg;
        }
        sg.xadj[li + 1] = sg.xadj[li] + deg;
    }
    sg.adjncy.resize(sg.xadj[nv]);
    sg.adjwgt.resize(sg.xadj[nv]);
    for (int li = 0; li < nv; ++li) {
        int gi = verts[li];
        int pos = sg.xadj[li];
        for (int j = g.xadj[gi]; j < g.xadj[gi + 1]; ++j) {
            auto it = g2l.find(g.adjncy[j]);
            if (it != g2l.end()) {
                sg.adjncy[pos] = it->second;
                sg.adjwgt[pos] = g.adjwgt[j];
                ++pos;
            }
        }
    }
    return sg;
}

void GraphPartitioner::recursive_bisect(const Graph& g, int n_parts,
                                         std::vector<int>& part_ids, int offset) {
    const int nv = g.n_vertices;
    if (n_parts == 1 || nv == 0) {
        for (int v = 0; v < nv; ++v) part_ids[v] = offset;
        return;
    }

    // Bisect
    std::vector<int> bisect = multilevel_bisect(g);

    const int left_parts  = n_parts / 2;
    const int right_parts = n_parts - left_parts;

    // Collect vertices in each half
    std::vector<int> left_verts, right_verts;
    for (int v = 0; v < nv; ++v) {
        if (bisect[v] == 0) left_verts.push_back(v);
        else                right_verts.push_back(v);
    }

    if (left_parts == 1) {
        for (int v : left_verts) part_ids[v] = offset;
    } else {
        std::vector<int> l2g;
        Graph sg = subgraph(g, left_verts, l2g);
        std::vector<int> sub_parts(static_cast<int>(left_verts.size()));
        recursive_bisect(sg, left_parts, sub_parts, 0);
        for (int li = 0; li < static_cast<int>(left_verts.size()); ++li)
            part_ids[left_verts[li]] = offset + sub_parts[li];
    }

    if (right_parts == 1) {
        for (int v : right_verts) part_ids[v] = offset + left_parts;
    } else {
        std::vector<int> l2g;
        Graph sg = subgraph(g, right_verts, l2g);
        std::vector<int> sub_parts(static_cast<int>(right_verts.size()));
        recursive_bisect(sg, right_parts, sub_parts, 0);
        for (int li = 0; li < static_cast<int>(right_verts.size()); ++li)
            part_ids[right_verts[li]] = offset + left_parts + sub_parts[li];
    }
}

std::vector<int> GraphPartitioner::partition(const FVMesh& mesh, int n_parts) {
    if (n_parts <= 0) throw std::invalid_argument("n_parts must be >= 1");

    std::vector<int> part_ids(mesh.n_cells, 0);
    if (n_parts == 1) return part_ids;

    Graph g = build_dual_graph(mesh);
    recursive_bisect(g, n_parts, part_ids, 0);
    return part_ids;
}

} // namespace twofluid
