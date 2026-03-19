#include "twofluid/mesh.hpp"

namespace twofluid {

FVMesh::FVMesh(int ndim) : ndim(ndim) {
    nodes.resize(0, 3);
}

void FVMesh::build_boundary_face_cache() {
    boundary_face_cache.clear();
    for (auto& [bname, fids] : boundary_patches) {
        for (int li = 0; li < (int)fids.size(); ++li) {
            boundary_face_cache[fids[li]] = {bname, li};
        }
    }
}

std::string FVMesh::summary() const {
    std::ostringstream ss;
    ss << "FVMesh(" << ndim << "D): " << n_cells << " cells, "
       << n_faces << " faces (" << n_internal_faces << " internal, "
       << n_boundary_faces << " boundary)\n";
    ss << "  Nodes: " << nodes.rows() << "\n";
    for (const auto& [name, fids] : boundary_patches) {
        ss << "  Boundary '" << name << "': " << fids.size() << " faces\n";
    }
    for (const auto& [name, cids] : cell_zones) {
        ss << "  Zone '" << name << "': " << cids.size() << " cells\n";
    }
    return ss.str();
}

} // namespace twofluid
