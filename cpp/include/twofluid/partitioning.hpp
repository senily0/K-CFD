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
