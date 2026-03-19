#include "twofluid/hybrid_mesh_generator.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace twofluid {

// Consistent quad→2 triangle split (min-node diagonal)
static std::vector<std::array<int,3>> split_quad_consistent(int a, int b, int c, int d) {
    int m = std::min({a, b, c, d});
    if (m == a || m == c)
        return {{a, b, c}, {a, c, d}};
    else
        return {{a, b, d}, {b, c, d}};
}

// Hex→12 tets via center point insertion
static std::vector<std::array<int,4>> hex_to_tets(const int* n, int cid) {
    // 6 faces of hex
    int faces[6][4] = {
        {n[0], n[3], n[2], n[1]},  // bottom
        {n[4], n[5], n[6], n[7]},  // top
        {n[0], n[1], n[5], n[4]},  // front
        {n[2], n[3], n[7], n[6]},  // back
        {n[0], n[4], n[7], n[3]},  // left
        {n[1], n[2], n[6], n[5]},  // right
    };
    std::vector<std::array<int,4>> tets;
    for (auto& f : faces) {
        auto tris = split_quad_consistent(f[0], f[1], f[2], f[3]);
        for (auto& tri : tris) {
            tets.push_back({tri[0], tri[1], tri[2], cid});
        }
    }
    return tets;
}

static std::tuple<int,int,int> face_key3(int a, int b, int c) {
    int arr[3] = {a, b, c};
    std::sort(arr, arr + 3);
    return {arr[0], arr[1], arr[2]};
}

static std::tuple<int,int,int,int> face_key4(int a, int b, int c, int d) {
    int arr[4] = {a, b, c, d};
    std::sort(arr, arr + 4);
    return {arr[0], arr[1], arr[2], arr[3]};
}

FVMesh generate_hybrid_hex_tet_mesh(
    double Lx, double Ly, double Lz,
    int nx, int ny, int nz,
    double tet_fraction,
    const std::unordered_map<std::string, std::string>& boundary_names_in) {

    std::unordered_map<std::string, std::string> bn = {
        {"x_min", "inlet"}, {"x_max", "outlet"},
        {"y_min", "wall_bottom"}, {"y_max", "wall_top"},
        {"z_min", "wall_front"}, {"z_max", "wall_back"}
    };
    for (auto& [k, v] : boundary_names_in) bn[k] = v;

    double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;
    int nx_hex = std::max(1, static_cast<int>(nx * (1.0 - tet_fraction)));
    int nx_tet = nx - nx_hex;

    int n_base = (nx + 1) * (ny + 1) * (nz + 1);
    int n_center = nx_tet * ny * nz;

    auto nid = [&](int i, int j, int k) {
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i;
    };

    FVMesh mesh(3);
    mesh.nodes.resize(n_base + n_center, 3);

    // Base nodes
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                int id = nid(i, j, k);
                mesh.nodes(id, 0) = i * dx;
                mesh.nodes(id, 1) = j * dy;
                mesh.nodes(id, 2) = k * dz;
            }

    // Center nodes for tet region
    std::map<std::tuple<int,int,int>, int> center_map;
    int cnode = n_base;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = nx_hex; i < nx; ++i) {
                mesh.nodes(cnode, 0) = (i + 0.5) * dx;
                mesh.nodes(cnode, 1) = (j + 0.5) * dy;
                mesh.nodes(cnode, 2) = (k + 0.5) * dz;
                center_map[{i, j, k}] = cnode;
                cnode++;
            }

    // Build cells
    std::vector<std::vector<int>> cell_nodes_list;
    std::vector<int> hex_zone, tet_zone;

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int n0 = nid(i,j,k), n1 = nid(i+1,j,k);
                int n2 = nid(i+1,j+1,k), n3 = nid(i,j+1,k);
                int n4 = nid(i,j,k+1), n5 = nid(i+1,j,k+1);
                int n6 = nid(i+1,j+1,k+1), n7 = nid(i,j+1,k+1);

                if (i < nx_hex) {
                    int ci = static_cast<int>(cell_nodes_list.size());
                    cell_nodes_list.push_back({n0,n1,n2,n3,n4,n5,n6,n7});
                    hex_zone.push_back(ci);
                } else {
                    int hn[8] = {n0,n1,n2,n3,n4,n5,n6,n7};
                    int cid = center_map[{i,j,k}];
                    auto tets = hex_to_tets(hn, cid);
                    for (auto& tet : tets) {
                        int ci = static_cast<int>(cell_nodes_list.size());
                        cell_nodes_list.push_back({tet[0], tet[1], tet[2], tet[3]});
                        tet_zone.push_back(ci);
                    }
                }
            }

    mesh.n_cells = static_cast<int>(cell_nodes_list.size());
    mesh.cells.resize(mesh.n_cells);

    for (int ci = 0; ci < mesh.n_cells; ++ci) {
        auto& cell = mesh.cells[ci];
        cell.nodes = cell_nodes_list[ci];
        // Compute center
        Vec3 c = Vec3::Zero();
        for (int nid_val : cell.nodes)
            c += mesh.nodes.row(nid_val).transpose();
        c /= cell.nodes.size();
        cell.center = c;

        // Volume: for hex = dx*dy*dz, for tet = |det|/6
        if (cell.nodes.size() == 8) {
            cell.volume = dx * dy * dz;
        } else if (cell.nodes.size() == 4) {
            Vec3 a = mesh.nodes.row(cell.nodes[1]).transpose()
                   - mesh.nodes.row(cell.nodes[0]).transpose();
            Vec3 b = mesh.nodes.row(cell.nodes[2]).transpose()
                   - mesh.nodes.row(cell.nodes[0]).transpose();
            Vec3 cv = mesh.nodes.row(cell.nodes[3]).transpose()
                    - mesh.nodes.row(cell.nodes[0]).transpose();
            cell.volume = std::abs(a.dot(b.cross(cv))) / 6.0;
        }
    }

    mesh.cell_zones["hex"] = hex_zone;
    mesh.cell_zones["tet"] = tet_zone;

    // Build faces using a face-key → cell mapping
    // For simplicity, use sorted-node tuples as face keys
    using FKey = std::vector<int>;
    auto make_key = [](std::vector<int> ns) {
        std::sort(ns.begin(), ns.end());
        return ns;
    };

    struct FaceInfo {
        std::vector<int> nodes;
        int first_cell = -1;
        int second_cell = -1;
    };
    std::map<FKey, FaceInfo> face_map;

    auto register_face = [&](const std::vector<int>& fnodes, int ci) {
        auto key = make_key(fnodes);
        auto it = face_map.find(key);
        if (it == face_map.end()) {
            face_map[key] = {fnodes, ci, -1};
        } else {
            it->second.second_cell = ci;
        }
    };

    for (int ci = 0; ci < mesh.n_cells; ++ci) {
        auto& cn = cell_nodes_list[ci];
        if (cn.size() == 8) {
            // Hex: 6 quad faces
            int n0=cn[0],n1=cn[1],n2=cn[2],n3=cn[3],n4=cn[4],n5=cn[5],n6=cn[6],n7=cn[7];
            register_face({n0,n3,n2,n1}, ci);
            register_face({n4,n5,n6,n7}, ci);
            register_face({n0,n1,n5,n4}, ci);
            register_face({n2,n3,n7,n6}, ci);
            register_face({n0,n4,n7,n3}, ci);
            register_face({n1,n2,n6,n5}, ci);
        } else if (cn.size() == 4) {
            // Tet: 4 tri faces
            register_face({cn[0],cn[2],cn[1]}, ci);
            register_face({cn[0],cn[1],cn[3]}, ci);
            register_face({cn[1],cn[2],cn[3]}, ci);
            register_face({cn[0],cn[3],cn[2]}, ci);
        }
    }

    // Create Face objects
    int fid = 0;
    double tol = 1e-10;
    for (auto& [key, fi] : face_map) {
        Face f;
        f.owner = fi.first_cell;
        f.neighbour = fi.second_cell;
        f.nodes = fi.nodes;

        // Geometry
        Vec3 center = Vec3::Zero();
        for (int nid_val : f.nodes)
            center += mesh.nodes.row(nid_val).transpose();
        center /= f.nodes.size();
        f.center = center;

        if (f.nodes.size() == 3) {
            // Triangle
            Vec3 a = mesh.nodes.row(f.nodes[1]).transpose()
                   - mesh.nodes.row(f.nodes[0]).transpose();
            Vec3 b = mesh.nodes.row(f.nodes[2]).transpose()
                   - mesh.nodes.row(f.nodes[0]).transpose();
            Vec3 n = a.cross(b);
            f.area = 0.5 * n.norm();
            f.normal = (n.norm() > 1e-30) ? (n / n.norm()).eval() : Vec3::Zero();
        } else if (f.nodes.size() == 4) {
            // Quad
            Vec3 d1 = mesh.nodes.row(f.nodes[2]).transpose()
                     - mesh.nodes.row(f.nodes[0]).transpose();
            Vec3 d2 = mesh.nodes.row(f.nodes[3]).transpose()
                     - mesh.nodes.row(f.nodes[1]).transpose();
            Vec3 n = d1.cross(d2);
            f.area = 0.5 * n.norm();
            f.normal = (n.norm() > 1e-30) ? (n / n.norm()).eval() : Vec3::Zero();
        }

        // Orient normal: owner→neighbour or outward
        if (f.neighbour >= 0) {
            Vec3 d = mesh.cells[f.neighbour].center - mesh.cells[f.owner].center;
            if (f.normal.dot(d) < 0) f.normal = -f.normal;
        } else {
            Vec3 d = f.center - mesh.cells[f.owner].center;
            if (f.normal.dot(d) < 0) f.normal = -f.normal;
        }

        mesh.cells[f.owner].faces.push_back(fid);
        if (f.neighbour >= 0) mesh.cells[f.neighbour].faces.push_back(fid);

        // Boundary classification
        if (f.neighbour < 0) {
            // Check if face is on a domain boundary (no center nodes)
            bool has_center = false;
            for (int nv : f.nodes)
                if (nv >= n_base) { has_center = true; break; }

            if (!has_center) {
                // Check which boundary
                bool all_xmin = true, all_xmax = true;
                bool all_ymin = true, all_ymax = true;
                bool all_zmin = true, all_zmax = true;
                for (int nv : f.nodes) {
                    double x = mesh.nodes(nv, 0), y = mesh.nodes(nv, 1), z = mesh.nodes(nv, 2);
                    if (std::abs(x) > tol) all_xmin = false;
                    if (std::abs(x - Lx) > tol) all_xmax = false;
                    if (std::abs(y) > tol) all_ymin = false;
                    if (std::abs(y - Ly) > tol) all_ymax = false;
                    if (std::abs(z) > tol) all_zmin = false;
                    if (std::abs(z - Lz) > tol) all_zmax = false;
                }
                std::string bname;
                if (all_xmin) bname = bn["x_min"];
                else if (all_xmax) bname = bn["x_max"];
                else if (all_ymin) bname = bn["y_min"];
                else if (all_ymax) bname = bn["y_max"];
                else if (all_zmin) bname = bn["z_min"];
                else if (all_zmax) bname = bn["z_max"];

                if (!bname.empty()) {
                    f.boundary_tag = bname;
                    mesh.boundary_patches[bname].push_back(fid);
                }
            }
        }

        mesh.faces.push_back(f);
        fid++;
    }

    mesh.n_faces = fid;
    mesh.n_internal_faces = 0;
    mesh.n_boundary_faces = 0;
    for (auto& f : mesh.faces) {
        if (f.neighbour >= 0) mesh.n_internal_faces++;
        else mesh.n_boundary_faces++;
    }

    return mesh;
}

} // namespace twofluid
