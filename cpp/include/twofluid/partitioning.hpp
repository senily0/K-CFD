#pragma once
#include <vector>
#include <unordered_map>
#include "twofluid/mesh.hpp"

namespace twofluid {

/// Recursive Coordinate Bisection (RCB) partitioner.
/// Splits mesh into n_parts domains based on cell center coordinates.
class RCBPartitioner {
public:
    /// Partition mesh into n_parts. Returns part_id per cell [0, n_parts).
    static std::vector<int> partition(const FVMesh& mesh, int n_parts);

private:
    static void bisect(const FVMesh& mesh, const std::vector<int>& cell_ids,
                       int n_parts, std::vector<int>& part_ids, int part_offset);
};

/// Graph-based multilevel k-way partitioner (METIS-like).
/// Builds the dual graph of the mesh (cells as vertices, shared faces as edges),
/// then applies multilevel recursive bisection with Kernighan-Lin refinement.
/// Minimises edge-cut (communication surface) compared to RCB.
class GraphPartitioner {
public:
    /// Partition mesh into n_parts using multilevel graph bisection.
    /// Returns part_id per cell [0, n_parts).
    static std::vector<int> partition(const FVMesh& mesh, int n_parts);

private:
    /// CSR adjacency representation of the dual graph.
    struct Graph {
        int n_vertices = 0;
        std::vector<int> xadj;    // row pointers (size n_vertices+1)
        std::vector<int> adjncy;  // column indices
        std::vector<int> adjwgt;  // edge weights (1 per shared face)
        // Cell-centre coordinates (for BFS seed selection and coarsening tie-breaks)
        std::vector<double> cx, cy, cz;
    };

    /// Build dual graph from mesh internal faces.
    static Graph build_dual_graph(const FVMesh& mesh);

    /// Coarsen graph by heavy-edge matching; returns coarsened graph and
    /// mapping coarse_id -> list of fine vertex ids (for uncoarsening).
    static Graph coarsen(const Graph& g, std::vector<std::vector<int>>& coarse_to_fine);

    /// Bisect graph into two parts; returns 0/1 assignment per vertex.
    /// Uses BFS from two antipodal seeds found by coordinate spread.
    static std::vector<int> initial_bisect(const Graph& g);

    /// Kernighan-Lin / Fiduccia-Mattheyses boundary refinement.
    /// Swaps boundary vertices to reduce edge-cut. Modifies parts in-place.
    static void kl_refine(const Graph& g, std::vector<int>& parts);

    /// Multilevel bisection: coarsen -> bisect coarsest -> uncoarsen + refine.
    static std::vector<int> multilevel_bisect(const Graph& g);

    /// Recursive bisection for n_parts > 2.
    /// part_ids is over the subgraph vertex indices (0..g.n_vertices-1),
    /// offset is added to all assigned part ids.
    static void recursive_bisect(const Graph& g, int n_parts,
                                  std::vector<int>& part_ids, int offset);

    /// Extract subgraph induced by vertex set; returns new graph and index map.
    static Graph subgraph(const Graph& g, const std::vector<int>& verts,
                          std::vector<int>& local_to_global);
};

/// Ghost cell layer management for distributed solve.
struct GhostLayer {
    std::vector<int> send_cells;  // local cells to send to neighbor
    std::vector<int> recv_cells;  // ghost cell indices (appended after local)
    int neighbor_rank;
};

/// Build ghost layers for a partitioned mesh.
/// Returns ghost layers per neighboring rank.
std::vector<GhostLayer> build_ghost_layers(
    const FVMesh& mesh, const std::vector<int>& part_ids, int my_rank);

/// Extract local submesh for given partition.
/// Returns (local_mesh, local_to_global_map, global_to_local_map)
struct LocalMesh {
    FVMesh mesh;
    std::vector<int> local_to_global;
    std::unordered_map<int, int> global_to_local;
};
LocalMesh extract_local_mesh(const FVMesh& global_mesh,
                              const std::vector<int>& part_ids, int my_rank);

} // namespace twofluid
