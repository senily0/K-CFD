#include "twofluid/mesh_generator_3d.hpp"
#include <cmath>
#include <map>

namespace twofluid {

FVMesh generate_3d_channel_mesh(
    double Lx, double Ly, double Lz,
    int nx, int ny, int nz,
    const std::unordered_map<std::string, std::string>& boundary_names_in) {

    // Default boundary names
    std::unordered_map<std::string, std::string> bn = {
        {"x_min", "inlet"}, {"x_max", "outlet"},
        {"y_min", "wall_bottom"}, {"y_max", "wall_top"},
        {"z_min", "wall_front"}, {"z_max", "wall_back"}
    };
    for (auto& [k, v] : boundary_names_in) bn[k] = v;

    double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;
    int n_nodes = (nx + 1) * (ny + 1) * (nz + 1);
    int n_cells = nx * ny * nz;

    auto nid = [&](int i, int j, int k) {
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
    };
    auto cid = [&](int i, int j, int k) {
        return k * ny * nx + j * nx + i;
    };

    FVMesh mesh(3);

    // Nodes
    mesh.nodes.resize(n_nodes, 3);
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                int id = nid(i, j, k);
                mesh.nodes(id, 0) = i * dx;
                mesh.nodes(id, 1) = j * dy;
                mesh.nodes(id, 2) = k * dz;
            }

    // Cells
    mesh.n_cells = n_cells;
    mesh.cells.resize(n_cells);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int ci = cid(i, j, k);
                Cell& cell = mesh.cells[ci];
                cell.nodes = {
                    nid(i, j, k), nid(i+1, j, k),
                    nid(i+1, j+1, k), nid(i, j+1, k),
                    nid(i, j, k+1), nid(i+1, j, k+1),
                    nid(i+1, j+1, k+1), nid(i, j+1, k+1)
                };
                cell.center = Vec3(
                    (i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz);
                cell.volume = dx * dy * dz;
            }

    // Helper: create a face between owner and neighbour
    int face_id = 0;
    auto add_face = [&](int owner, int nb,
                        int n0, int n1, int n2, int n3,
                        const Vec3& normal_dir) {
        Face f;
        f.owner = owner;
        f.neighbour = nb;
        f.nodes = {n0, n1, n2, n3};

        Vec3 p0 = mesh.nodes.row(n0).transpose();
        Vec3 p1 = mesh.nodes.row(n1).transpose();
        Vec3 p2 = mesh.nodes.row(n2).transpose();
        Vec3 p3 = mesh.nodes.row(n3).transpose();
        f.center = 0.25 * (p0 + p1 + p2 + p3);

        // Area = |d1 x d2| for quad (two triangles)
        Vec3 d1 = p2 - p0;
        Vec3 d2 = p3 - p1;
        Vec3 cross = d1.cross(d2);
        f.area = 0.5 * cross.norm();
        f.normal = normal_dir;

        mesh.cells[owner].faces.push_back(face_id);
        if (nb >= 0) mesh.cells[nb].faces.push_back(face_id);
        mesh.faces.push_back(f);
        face_id++;
    };

    // Internal x-faces (between i and i+1)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx - 1; ++i) {
                int owner = cid(i, j, k);
                int nb = cid(i + 1, j, k);
                add_face(owner, nb,
                    nid(i+1, j, k), nid(i+1, j+1, k),
                    nid(i+1, j+1, k+1), nid(i+1, j, k+1),
                    Vec3(1, 0, 0));
            }

    // Internal y-faces
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny - 1; ++j)
            for (int i = 0; i < nx; ++i) {
                int owner = cid(i, j, k);
                int nb = cid(i, j + 1, k);
                add_face(owner, nb,
                    nid(i, j+1, k), nid(i+1, j+1, k),
                    nid(i+1, j+1, k+1), nid(i, j+1, k+1),
                    Vec3(0, 1, 0));
            }

    // Internal z-faces
    for (int k = 0; k < nz - 1; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int owner = cid(i, j, k);
                int nb = cid(i, j, k + 1);
                add_face(owner, nb,
                    nid(i, j, k+1), nid(i+1, j, k+1),
                    nid(i+1, j+1, k+1), nid(i, j+1, k+1),
                    Vec3(0, 0, 1));
            }

    // Boundary faces: x=0 (inlet)
    std::string bx0 = bn["x_min"];
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            int owner = cid(0, j, k);
            add_face(owner, -1,
                nid(0, j, k), nid(0, j, k+1),
                nid(0, j+1, k+1), nid(0, j+1, k),
                Vec3(-1, 0, 0));
            mesh.faces.back().boundary_tag = bx0;
            mesh.boundary_patches[bx0].push_back(face_id - 1);
        }

    // x=Lx (outlet)
    std::string bx1 = bn["x_max"];
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            int owner = cid(nx - 1, j, k);
            add_face(owner, -1,
                nid(nx, j, k), nid(nx, j+1, k),
                nid(nx, j+1, k+1), nid(nx, j, k+1),
                Vec3(1, 0, 0));
            mesh.faces.back().boundary_tag = bx1;
            mesh.boundary_patches[bx1].push_back(face_id - 1);
        }

    // y=0 (wall_bottom)
    std::string by0 = bn["y_min"];
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i) {
            int owner = cid(i, 0, k);
            add_face(owner, -1,
                nid(i, 0, k), nid(i+1, 0, k),
                nid(i+1, 0, k+1), nid(i, 0, k+1),
                Vec3(0, -1, 0));
            mesh.faces.back().boundary_tag = by0;
            mesh.boundary_patches[by0].push_back(face_id - 1);
        }

    // y=Ly (wall_top)
    std::string by1 = bn["y_max"];
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i) {
            int owner = cid(i, ny - 1, k);
            add_face(owner, -1,
                nid(i, ny, k), nid(i, ny, k+1),
                nid(i+1, ny, k+1), nid(i+1, ny, k),
                Vec3(0, 1, 0));
            mesh.faces.back().boundary_tag = by1;
            mesh.boundary_patches[by1].push_back(face_id - 1);
        }

    // z=0 (wall_front)
    std::string bz0 = bn["z_min"];
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int owner = cid(i, j, 0);
            add_face(owner, -1,
                nid(i, j, 0), nid(i, j+1, 0),
                nid(i+1, j+1, 0), nid(i+1, j, 0),
                Vec3(0, 0, -1));
            mesh.faces.back().boundary_tag = bz0;
            mesh.boundary_patches[bz0].push_back(face_id - 1);
        }

    // z=Lz (wall_back)
    std::string bz1 = bn["z_max"];
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int owner = cid(i, j, nz - 1);
            add_face(owner, -1,
                nid(i, j, nz), nid(i+1, j, nz),
                nid(i+1, j+1, nz), nid(i, j+1, nz),
                Vec3(0, 0, 1));
            mesh.faces.back().boundary_tag = bz1;
            mesh.boundary_patches[bz1].push_back(face_id - 1);
        }

    mesh.n_faces = face_id;
    mesh.n_internal_faces = 0;
    mesh.n_boundary_faces = 0;
    for (const auto& f : mesh.faces) {
        if (f.neighbour >= 0) mesh.n_internal_faces++;
        else mesh.n_boundary_faces++;
    }

    mesh.build_boundary_face_cache();
    return mesh;
}

FVMesh generate_3d_duct_mesh(double Lx, double Ly, double Lz,
                              int nx, int ny, int nz) {
    return generate_3d_channel_mesh(Lx, Ly, Lz, nx, ny, nz, {
        {"x_min", "inlet"}, {"x_max", "outlet"},
        {"y_min", "wall_bottom"}, {"y_max", "wall_top"},
        {"z_min", "wall_front"}, {"z_max", "wall_back"}
    });
}

FVMesh generate_3d_cavity_mesh(double Lx, double Ly, double Lz,
                                int nx, int ny, int nz) {
    return generate_3d_channel_mesh(Lx, Ly, Lz, nx, ny, nz, {
        {"x_min", "wall_left"}, {"x_max", "wall_right"},
        {"y_min", "wall_bottom"}, {"y_max", "lid"},
        {"z_min", "wall_front"}, {"z_max", "wall_back"}
    });
}

} // namespace twofluid
