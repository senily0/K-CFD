#pragma once

#include <string>
#include <unordered_map>
#include "twofluid/mesh.hpp"

namespace twofluid {

/// Generate a hybrid Hex/Tet 3D mesh.
/// x < (1-tet_fraction)*Lx uses hexahedra, remainder uses tetrahedra
/// via center-point insertion (12 tets per hex).
FVMesh generate_hybrid_hex_tet_mesh(
    double Lx = 2.0, double Ly = 0.1, double Lz = 0.1,
    int nx = 20, int ny = 8, int nz = 8,
    double tet_fraction = 0.5,
    const std::unordered_map<std::string, std::string>& boundary_names = {});

} // namespace twofluid
