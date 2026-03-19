#include "twofluid/mesh_generator.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace twofluid {
namespace {

/// Edge key: sorted pair of node indices.
using EdgeKey = std::pair<int, int>;

EdgeKey edge_key(int n0, int n1) {
    return (n0 < n1) ? EdgeKey{n0, n1} : EdgeKey{n1, n0};
}

/// 2D face geometry: center, outward unit normal, length.
struct FaceGeo {
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    double length;
};

FaceGeo compute_face_geometry_2d(const Eigen::MatrixXd& nodes, int n0, int n1) {
    FaceGeo g;
    double x0 = nodes(n0, 0), y0 = nodes(n0, 1);
    double x1 = nodes(n1, 0), y1 = nodes(n1, 1);
    g.center = Eigen::Vector3d(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.0);
    double dx = x1 - x0;
    double dy = y1 - y0;
    g.length = std::sqrt(dx * dx + dy * dy);
    // Outward normal (right-hand side of edge direction)
    g.normal = Eigen::Vector3d(dy, -dx, 0.0);
    if (g.length > 1e-15) {
        g.normal /= g.length;
    }
    return g;
}

/// 2D cell geometry: center, area (shoelace formula).
struct CellGeo {
    Eigen::Vector3d center;
    double area;
};

CellGeo compute_cell_geometry_2d(const Eigen::MatrixXd& nodes,
                                  const std::vector<int>& cnodes) {
    CellGeo g;
    int n = static_cast<int>(cnodes.size());
    double cx = 0.0, cy = 0.0;
    for (int i = 0; i < n; ++i) {
        cx += nodes(cnodes[i], 0);
        cy += nodes(cnodes[i], 1);
    }
    g.center = Eigen::Vector3d(cx / n, cy / n, 0.0);

    double area = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += nodes(cnodes[i], 0) * nodes(cnodes[j], 1)
              - nodes(cnodes[j], 0) * nodes(cnodes[i], 1);
    }
    g.area = std::abs(area) / 2.0;
    return g;
}

/// Boundary face specification: node pair + boundary name.
struct BFaceSpec {
    int n0, n1;
    std::string name;
};

/// Core structured quad mesh builder (mirrors Python _make_structured_quad_mesh).
/// Returns nodes (2D), cell node lists, and boundary face specs.
struct RawMesh {
    Eigen::MatrixXd nodes;              // (n_nodes, 3)
    std::vector<std::vector<int>> cells; // cell node lists (CCW quads)
    std::vector<BFaceSpec> bfaces;
};

RawMesh make_structured_quad_mesh(
    double x0, double y0, double x1, double y1,
    int nx, int ny,
    const std::string& name_bottom,
    const std::string& name_top,
    const std::string& name_left,
    const std::string& name_right)
{
    RawMesh raw;
    int n_nodes = (nx + 1) * (ny + 1);
    raw.nodes.resize(n_nodes, 3);
    raw.nodes.setZero();

    double dx = (x1 - x0) / nx;
    double dy = (y1 - y0) / ny;

    // Create nodes
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            int nid = j * (nx + 1) + i;
            raw.nodes(nid, 0) = x0 + i * dx;
            raw.nodes(nid, 1) = y0 + j * dy;
            // z stays 0
        }
    }

    // Create cells (quad: 4 nodes, counter-clockwise)
    raw.cells.reserve(nx * ny);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int n0 = j * (nx + 1) + i;
            int n1 = n0 + 1;
            int n2 = n1 + (nx + 1);
            int n3 = n0 + (nx + 1);
            raw.cells.push_back({n0, n1, n2, n3});
        }
    }

    // Boundary faces
    // Bottom (j=0)
    for (int i = 0; i < nx; ++i) {
        raw.bfaces.push_back({i, i + 1, name_bottom});
    }
    // Top (j=ny)
    for (int i = 0; i < nx; ++i) {
        int n0 = ny * (nx + 1) + i;
        raw.bfaces.push_back({n0, n0 + 1, name_top});
    }
    // Left (i=0)
    for (int j = 0; j < ny; ++j) {
        int n0 = j * (nx + 1);
        int n1 = (j + 1) * (nx + 1);
        raw.bfaces.push_back({n0, n1, name_left});
    }
    // Right (i=nx)
    for (int j = 0; j < ny; ++j) {
        int n0 = j * (nx + 1) + nx;
        int n1 = (j + 1) * (nx + 1) + nx;
        raw.bfaces.push_back({n0, n1, name_right});
    }

    return raw;
}

/// Build a complete FVMesh from raw mesh data (mirrors build_fvmesh_from_arrays 2D path).
FVMesh build_fvmesh(RawMesh& raw) {
    FVMesh mesh(2);
    mesh.nodes = raw.nodes;

    int n_cells = static_cast<int>(raw.cells.size());

    // 1) Create cells with geometry
    mesh.cells.resize(n_cells);
    for (int ci = 0; ci < n_cells; ++ci) {
        auto& cell = mesh.cells[ci];
        cell.nodes = raw.cells[ci];
        auto geo = compute_cell_geometry_2d(mesh.nodes, cell.nodes);
        cell.center = geo.center;
        cell.volume = geo.area;
    }
    mesh.n_cells = n_cells;

    // 2) Extract edges, build faces with owner/neighbour
    std::map<EdgeKey, int> edge_to_face;

    for (int ci = 0; ci < n_cells; ++ci) {
        const auto& cnodes = mesh.cells[ci].nodes;
        int nn = static_cast<int>(cnodes.size());
        for (int k = 0; k < nn; ++k) {
            int n0 = cnodes[k];
            int n1 = cnodes[(k + 1) % nn];
            EdgeKey ek = edge_key(n0, n1);
            auto it = edge_to_face.find(ek);
            if (it != edge_to_face.end()) {
                // Second cell sharing this edge
                int fid = it->second;
                mesh.faces[fid].neighbour = ci;
                mesh.cells[ci].faces.push_back(fid);
            } else {
                // New face
                Face face;
                face.nodes = {n0, n1};
                face.owner = ci;
                face.neighbour = -1;
                auto geo = compute_face_geometry_2d(mesh.nodes, n0, n1);
                face.center = geo.center;
                face.area = geo.length;
                face.normal = geo.normal;
                // Ensure normal points outward from owner cell
                Eigen::Vector3d to_face = face.center - mesh.cells[ci].center;
                if (face.normal.dot(to_face) < 0) {
                    face.normal = -face.normal;
                }
                int fid = static_cast<int>(mesh.faces.size());
                mesh.faces.push_back(face);
                edge_to_face[ek] = fid;
                mesh.cells[ci].faces.push_back(fid);
            }
        }
    }

    // 3) Tag boundary faces
    std::map<EdgeKey, std::string> boundary_edge_map;
    for (const auto& bf : raw.bfaces) {
        boundary_edge_map[edge_key(bf.n0, bf.n1)] = bf.name;
    }

    for (int fid = 0; fid < static_cast<int>(mesh.faces.size()); ++fid) {
        auto& face = mesh.faces[fid];
        if (face.neighbour == -1) {
            EdgeKey ek = edge_key(face.nodes[0], face.nodes[1]);
            auto it = boundary_edge_map.find(ek);
            std::string bname = (it != boundary_edge_map.end()) ? it->second : "default";
            face.boundary_tag = bname;
            mesh.boundary_patches[bname].push_back(fid);
        }
    }

    // Counts
    mesh.n_faces = static_cast<int>(mesh.faces.size());
    mesh.n_internal_faces = 0;
    for (const auto& f : mesh.faces) {
        if (f.neighbour != -1) ++mesh.n_internal_faces;
    }
    mesh.n_boundary_faces = mesh.n_faces - mesh.n_internal_faces;

    return mesh;
}

} // anonymous namespace

// ---- Public API ----

FVMesh generate_channel_mesh(double Lx, double Ly, int nx, int ny) {
    auto raw = make_structured_quad_mesh(
        0.0, 0.0, Lx, Ly, nx, ny,
        "wall_bottom", "wall_top", "inlet", "outlet");
    auto mesh = build_fvmesh(raw);
    mesh.build_boundary_face_cache();
    return mesh;
}

FVMesh generate_cavity_mesh(double L, int n) {
    auto raw = make_structured_quad_mesh(
        0.0, 0.0, L, L, n, n,
        "wall_bottom", "lid", "wall_left", "wall_right");
    auto mesh = build_fvmesh(raw);
    mesh.build_boundary_face_cache();
    return mesh;
}

FVMesh generate_bfs_mesh(
    double step_height, double expansion_ratio,
    double L_up, double L_down,
    int nx_up, int nx_down, int ny)
{
    double H = step_height * expansion_ratio;
    double L_total = L_up + L_down;
    int nx = nx_up + nx_down;

    // Build full rectangle with temporary boundary names
    auto raw = make_structured_quad_mesh(
        0.0, 0.0, L_total, H, nx, ny,
        "_bottom_temp", "wall_top", "_left_temp", "outlet");

    // Split bottom boundary: x < L_up -> wall_step_top, x >= L_up -> wall_bottom
    // Split left boundary: y > step_height -> inlet, y <= step_height -> wall_step_inlet
    std::vector<BFaceSpec> new_bfaces;
    new_bfaces.reserve(raw.bfaces.size());

    for (const auto& bf : raw.bfaces) {
        if (bf.name == "_bottom_temp") {
            double xc = 0.5 * (raw.nodes(bf.n0, 0) + raw.nodes(bf.n1, 0));
            if (xc < L_up - 1e-10) {
                new_bfaces.push_back({bf.n0, bf.n1, "wall_step_top"});
            } else {
                new_bfaces.push_back({bf.n0, bf.n1, "wall_bottom"});
            }
        } else if (bf.name == "_left_temp") {
            double yc = 0.5 * (raw.nodes(bf.n0, 1) + raw.nodes(bf.n1, 1));
            if (yc > step_height + 1e-10) {
                new_bfaces.push_back({bf.n0, bf.n1, "inlet"});
            } else {
                new_bfaces.push_back({bf.n0, bf.n1, "wall_step_inlet"});
            }
        } else {
            new_bfaces.push_back(bf);
        }
    }
    raw.bfaces = std::move(new_bfaces);

    auto mesh = build_fvmesh(raw);
    mesh.build_boundary_face_cache();
    return mesh;
}

} // namespace twofluid
