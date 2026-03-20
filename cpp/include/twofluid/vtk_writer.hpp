#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"

namespace twofluid {

/// Write mesh and field data to VTU (VTK Unstructured Grid XML) format.
void write_vtu(
    const std::string& filename,
    const FVMesh& mesh,
    const std::unordered_map<std::string, Eigen::VectorXd>& cell_scalar_data = {},
    const std::unordered_map<std::string, Eigen::MatrixXd>& cell_vector_data = {}
);

/// Write VTU in binary format (appended raw binary, much smaller files).
void write_vtu_binary(
    const std::string& filename,
    const FVMesh& mesh,
    const std::unordered_map<std::string, Eigen::VectorXd>& cell_scalar_data = {},
    const std::unordered_map<std::string, Eigen::MatrixXd>& cell_vector_data = {}
);

} // namespace twofluid
