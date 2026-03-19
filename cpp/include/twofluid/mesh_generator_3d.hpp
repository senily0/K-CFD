#pragma once

#include <string>
#include <unordered_map>
#include "twofluid/mesh.hpp"

namespace twofluid {

FVMesh generate_3d_channel_mesh(
    double Lx = 1.0, double Ly = 0.1, double Lz = 0.1,
    int nx = 20, int ny = 10, int nz = 10,
    const std::unordered_map<std::string, std::string>& boundary_names = {});

FVMesh generate_3d_duct_mesh(
    double Lx = 2.0, double Ly = 0.1, double Lz = 0.1,
    int nx = 20, int ny = 10, int nz = 10);

FVMesh generate_3d_cavity_mesh(
    double Lx = 1.0, double Ly = 1.0, double Lz = 1.0,
    int nx = 16, int ny = 16, int nz = 16);

} // namespace twofluid
