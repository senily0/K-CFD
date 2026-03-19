#pragma once
#include <string>
#include "twofluid/mesh.hpp"

namespace twofluid {

/// Read a GMSH .msh file (format 2.2 ASCII) and build an FVMesh.
/// Physical groups from the $PhysicalNames section become boundary patch names.
/// Dimension is set to 2 if all z-coordinates are zero, otherwise 3.
FVMesh read_gmsh_msh(const std::string& filename);

} // namespace twofluid
