#pragma once
#include <string>
#include "twofluid/mesh.hpp"

namespace twofluid {

/// Read an OpenFOAM polyMesh directory and build an FVMesh.
/// path should point to the case directory (containing constant/polyMesh/).
/// Handles arbitrary polyhedral cells.
FVMesh read_openfoam_polymesh(const std::string& case_dir);

} // namespace twofluid
