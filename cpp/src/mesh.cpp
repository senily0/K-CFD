#include "twofluid/mesh.hpp"

namespace twofluid {

FVMesh::FVMesh(int ndim) : ndim(ndim) {
    nodes.resize(0, 3);
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
