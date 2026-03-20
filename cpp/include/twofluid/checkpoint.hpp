#pragma once
#include <string>
#include <utility>
#include <vector>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/two_fluid_solver.hpp"

namespace twofluid {

/// Write solver state to binary checkpoint file.
/// Format: header (magic + version + n_cells + n_fields) + field data
void write_checkpoint(const std::string& filename,
                       const FVMesh& mesh,
                       const std::vector<const ScalarField*>& scalar_fields,
                       const std::vector<const VectorField*>& vector_fields,
                       double time = 0.0, int step = 0);

/// Read checkpoint and populate fields.
/// Returns {time, step}. Fields must already be allocated with correct size.
std::pair<double, int> read_checkpoint(
    const std::string& filename,
    const FVMesh& mesh,
    std::vector<ScalarField*>& scalar_fields,
    std::vector<VectorField*>& vector_fields);

/// Convenience: write TwoFluidSolver state
void write_two_fluid_checkpoint(const std::string& filename,
                                  TwoFluidSolver& solver,
                                  double time, int step);

/// Convenience: read TwoFluidSolver state
std::pair<double, int> read_two_fluid_checkpoint(
    const std::string& filename, TwoFluidSolver& solver);

} // namespace twofluid
