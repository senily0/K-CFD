#pragma once

#include "twofluid/mesh.hpp"

namespace twofluid {

/// Generate a structured quad channel mesh.
FVMesh generate_channel_mesh(double Lx, double Ly, int nx, int ny);

/// Generate a lid-driven cavity mesh.
FVMesh generate_cavity_mesh(double L, int n);

/// Generate a backward-facing step mesh.
FVMesh generate_bfs_mesh(
    double step_height = 1.0, double expansion_ratio = 2.0,
    double L_up = 5.0, double L_down = 30.0,
    int nx_up = 50, int nx_down = 250, int ny = 80
);

} // namespace twofluid
