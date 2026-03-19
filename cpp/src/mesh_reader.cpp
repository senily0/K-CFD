/// GMSH .msh format 2.2 ASCII reader
/// Reference: http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format-version-2
#include "twofluid/mesh_reader.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace twofluid {

// ---------------------------------------------------------------------------
// GMSH element type -> number of nodes
// ---------------------------------------------------------------------------
static int gmsh_nodes_per_elem(int type) {
    switch (type) {
        case  1: return 2;   // 2-node line
        case  2: return 3;   // 3-node triangle
        case  3: return 4;   // 4-node quadrangle
        case  4: return 4;   // 4-node tetrahedron
        case  5: return 8;   // 8-node hexahedron
        case  6: return 6;   // 6-node prism
        case  7: return 5;   // 5-node pyramid
        case 15: return 1;   // 1-node point
        default: return -1;  // unknown
    }
}

// Is the element type a volume cell (2D face or 3D cell)?
static bool is_volume_elem(int type) {
    return type == 2 || type == 3 || type == 4 || type == 5 || type == 6 || type == 7;
}

// Is the element type a boundary element (lower-dimensional)?
static bool is_boundary_elem(int type) {
    return type == 1 || type == 2 || type == 3;
    // In a 3D mesh, triangles/quads are boundaries; in 2D mesh, lines are.
    // We distinguish later by mesh dimension.
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
static Vec3 cross3(const Vec3& a, const Vec3& b) {
    return Vec3(a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]);
}

// Compute face area, outward normal, and center for a polygonal face.
// Nodes given as rows of `pts` in order.  Owner cell center used to orient normal.
static void compute_face_geometry(const std::vector<Vec3>& pts,
                                  const Vec3& owner_center,
                                  double& area, Vec3& normal, Vec3& center)
{
    int n = static_cast<int>(pts.size());
    // Center as average of vertices
    center = Vec3::Zero();
    for (const auto& p : pts) center += p;
    center /= static_cast<double>(n);

    // Area and normal via fan triangulation from center
    Vec3 n_sum = Vec3::Zero();
    for (int i = 0; i < n; ++i) {
        Vec3 a = pts[i]            - center;
        Vec3 b = pts[(i+1) % n]   - center;
        n_sum += cross3(a, b);
    }
    area = 0.5 * n_sum.norm();
    if (area < 1e-300) {
        normal = Vec3(0.0, 0.0, 1.0);
        return;
    }
    normal = n_sum / n_sum.norm();

    // Orient normal away from owner cell
    if (normal.dot(center - owner_center) < 0.0)
        normal = -normal;
}

// Compute cell volume and center (polyhedral — fan from cell centroid)
static void compute_cell_geometry(const std::vector<Vec3>& pts,
                                  double& volume, Vec3& center)
{
    int n = static_cast<int>(pts.size());
    center = Vec3::Zero();
    for (const auto& p : pts) center += p;
    center /= static_cast<double>(n);

    // For 2D: area via shoelace in XY plane
    // For 3D: sum of tet volumes from center to each triangular face
    // Use a generic approach: sum signed volumes of tetrahedra fan from center.
    volume = 0.0;
    if (n == 3) {
        // Triangle: 0.5 * |cross(e1, e2)|
        Vec3 e1 = pts[1] - pts[0];
        Vec3 e2 = pts[2] - pts[0];
        volume = 0.5 * cross3(e1, e2).norm();
    } else if (n == 4) {
        // Quad (2D) or Tet (3D)?
        // Check if coplanar (z == 0 for all -> 2D quad)
        bool flat = true;
        for (const auto& p : pts)
            if (std::abs(p[2]) > 1e-12) { flat = false; break; }
        if (flat) {
            // Shoelace in XY
            double s = 0.0;
            for (int i = 0; i < n; ++i) {
                int j = (i + 1) % n;
                s += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
            }
            volume = 0.5 * std::abs(s);
        } else {
            // Tetrahedron
            Vec3 e1 = pts[1] - pts[0];
            Vec3 e2 = pts[2] - pts[0];
            Vec3 e3 = pts[3] - pts[0];
            volume = std::abs(e1.dot(cross3(e2, e3))) / 6.0;
        }
    } else if (n == 8) {
        // Hexahedron: split into 5 tets
        // Indices for 5-tet decomposition of a hex
        static const int tet[5][4] = {
            {0,1,3,4}, {1,2,3,6}, {3,4,6,7}, {1,4,5,6}, {1,3,4,6}
        };
        volume = 0.0;
        for (auto& t : tet) {
            Vec3 e1 = pts[t[1]] - pts[t[0]];
            Vec3 e2 = pts[t[2]] - pts[t[0]];
            Vec3 e3 = pts[t[3]] - pts[t[0]];
            volume += std::abs(e1.dot(cross3(e2, e3))) / 6.0;
        }
    } else if (n == 6) {
        // Prism: split into 3 tets
        static const int tet[3][4] = {
            {0,1,2,3}, {1,3,4,2}, {2,3,4,5} // corrected decomposition
        };
        volume = 0.0;
        for (auto& t : tet) {
            Vec3 e1 = pts[t[1]] - pts[t[0]];
            Vec3 e2 = pts[t[2]] - pts[t[0]];
            Vec3 e3 = pts[t[3]] - pts[t[0]];
            volume += std::abs(e1.dot(cross3(e2, e3))) / 6.0;
        }
    } else if (n == 5) {
        // Pyramid: split into 2 tets
        static const int tet[2][4] = {{0,1,3,4},{1,2,3,4}};
        volume = 0.0;
        for (auto& t : tet) {
            Vec3 e1 = pts[t[1]] - pts[t[0]];
            Vec3 e2 = pts[t[2]] - pts[t[0]];
            Vec3 e3 = pts[t[3]] - pts[t[0]];
            volume += std::abs(e1.dot(cross3(e2, e3))) / 6.0;
        }
    } else {
        // Generic: fan triangulation from center, sum tet volumes
        for (int i = 0; i < n - 2; ++i) {
            Vec3 e1 = pts[i+1] - pts[0];
            Vec3 e2 = pts[i+2] - pts[0];
            volume += 0.5 * cross3(e1, e2).norm();
        }
    }
    if (volume < 0.0) volume = -volume;
}

// Extract the face node-sets for each element type (local face connectivity)
// Returns list of faces, each as a sorted set of global node ids.
static std::vector<std::vector<int>> cell_face_nodes(int elem_type,
                                                      const std::vector<int>& cell_nodes)
{
    // Local face connectivity (0-based indices into cell_nodes)
    std::vector<std::vector<int>> local_faces;
    switch (elem_type) {
        case 2: // triangle: 3 edges
            local_faces = {{0,1},{1,2},{2,0}};
            break;
        case 3: // quad: 4 edges
            local_faces = {{0,1},{1,2},{2,3},{3,0}};
            break;
        case 4: // tet: 4 triangular faces
            local_faces = {{0,1,2},{0,1,3},{1,2,3},{0,2,3}};
            break;
        case 5: // hex: 6 quad faces
            local_faces = {{0,1,2,3},{4,5,6,7},{0,1,5,4},
                           {1,2,6,5},{2,3,7,6},{3,0,4,7}};
            break;
        case 6: // prism: 2 tri + 3 quad
            local_faces = {{0,1,2},{3,4,5},{0,1,4,3},{1,2,5,4},{2,0,3,5}};
            break;
        case 7: // pyramid: 1 quad + 4 tri
            local_faces = {{0,1,2,3},{0,1,4},{1,2,4},{2,3,4},{3,0,4}};
            break;
        default:
            break;
    }
    std::vector<std::vector<int>> result;
    result.reserve(local_faces.size());
    for (auto& lf : local_faces) {
        std::vector<int> gf;
        gf.reserve(lf.size());
        for (int li : lf) gf.push_back(cell_nodes[li]);
        result.push_back(std::move(gf));
    }
    return result;
}

// Canonical face key: sorted node list as a string
static std::string face_key(std::vector<int> nodes) {
    std::sort(nodes.begin(), nodes.end());
    std::string k;
    k.reserve(nodes.size() * 6);
    for (int n : nodes) { k += std::to_string(n); k += ','; }
    return k;
}

// ---------------------------------------------------------------------------
// Main reader
// ---------------------------------------------------------------------------
FVMesh read_gmsh_msh(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("read_gmsh_msh: cannot open '" + filename + "'");

    // Raw storage from file
    std::map<int, std::string> phys_names;          // physical_tag -> name
    std::vector<Vec3>          raw_nodes;            // indexed by (node_id - 1)
    std::map<int, int>         node_id_map;          // gmsh 1-based id -> 0-based index
    int                        max_node_id = 0;

    struct GmshElem {
        int type;
        int phys_tag;   // first tag (physical group)
        int geom_tag;   // second tag (elementary entity)
        std::vector<int> nodes;  // 0-based
    };
    std::vector<GmshElem> elems;

    std::string line;
    while (std::getline(f, line)) {
        if (line == "$MeshFormat") {
            // Read and verify format line
            std::getline(f, line);  // "2.2 0 8"
            double ver; int type_f, size_f;
            std::istringstream ss(line);
            ss >> ver >> type_f >> size_f;
            if (ver < 2.0 || ver >= 3.0 || type_f != 0)
                throw std::runtime_error("read_gmsh_msh: only ASCII format 2.x supported");
            std::getline(f, line);  // $EndMeshFormat
        }
        else if (line == "$PhysicalNames") {
            int nphys = 0;
            std::getline(f, line);
            std::istringstream ss(line);
            ss >> nphys;
            for (int i = 0; i < nphys; ++i) {
                std::getline(f, line);
                std::istringstream ls(line);
                int dim, tag;
                std::string name;
                ls >> dim >> tag >> name;
                // Remove surrounding quotes if present
                if (!name.empty() && name.front() == '"') name = name.substr(1);
                if (!name.empty() && name.back()  == '"') name.pop_back();
                phys_names[tag] = name;
            }
            std::getline(f, line);  // $EndPhysicalNames
        }
        else if (line == "$Nodes") {
            int n_nodes = 0;
            std::getline(f, line);
            std::istringstream ss(line);
            ss >> n_nodes;
            raw_nodes.reserve(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                std::getline(f, line);
                std::istringstream ls(line);
                int id;
                double x, y, z;
                ls >> id >> x >> y >> z;
                int idx = static_cast<int>(raw_nodes.size());
                node_id_map[id] = idx;
                raw_nodes.push_back(Vec3(x, y, z));
                if (id > max_node_id) max_node_id = id;
            }
            std::getline(f, line);  // $EndNodes
        }
        else if (line == "$Elements") {
            int n_elems = 0;
            std::getline(f, line);
            std::istringstream ss(line);
            ss >> n_elems;
            elems.reserve(n_elems);
            for (int i = 0; i < n_elems; ++i) {
                std::getline(f, line);
                std::istringstream ls(line);
                int id, type, n_tags;
                ls >> id >> type >> n_tags;
                int phys_tag = 0, geom_tag = 0;
                for (int t = 0; t < n_tags; ++t) {
                    int tag_val; ls >> tag_val;
                    if (t == 0) phys_tag = tag_val;
                    if (t == 1) geom_tag = tag_val;
                }
                int npts = gmsh_nodes_per_elem(type);
                if (npts < 0) continue;  // unsupported type
                std::vector<int> enodes(npts);
                for (int k = 0; k < npts; ++k) {
                    int nid; ls >> nid;
                    enodes[k] = node_id_map.count(nid) ? node_id_map[nid] : (nid - 1);
                }
                elems.push_back({type, phys_tag, geom_tag, std::move(enodes)});
            }
            std::getline(f, line);  // $EndElements
        }
    }
    f.close();

    // -----------------------------------------------------------------------
    // Detect mesh dimension from z-coordinates
    // -----------------------------------------------------------------------
    bool has_z = false;
    for (const auto& p : raw_nodes)
        if (std::abs(p[2]) > 1e-12) { has_z = true; break; }
    int ndim = has_z ? 3 : 2;

    FVMesh mesh(ndim);

    // Copy nodes into Eigen matrix
    int n_nodes = static_cast<int>(raw_nodes.size());
    mesh.nodes.resize(n_nodes, 3);
    for (int i = 0; i < n_nodes; ++i)
        mesh.nodes.row(i) = raw_nodes[i].transpose();

    // -----------------------------------------------------------------------
    // Separate volume cells from boundary surface elements
    // For 2D: volume = tri/quad, boundary = lines
    // For 3D: volume = tet/hex/prism/pyramid, boundary = tri/quad
    // -----------------------------------------------------------------------
    auto is_vol = [&](int type) -> bool {
        if (ndim == 2) return type == 2 || type == 3;
        else           return type == 4 || type == 5 || type == 6 || type == 7;
    };
    auto is_bnd = [&](int type) -> bool {
        if (ndim == 2) return type == 1;
        else           return type == 2 || type == 3;
    };

    // Build cells
    struct RawCell { int type; int phys_tag; std::vector<int> nodes; };
    std::vector<RawCell> raw_cells;
    for (auto& e : elems)
        if (is_vol(e.type))
            raw_cells.push_back({e.type, e.phys_tag, e.nodes});

    // Build boundary surface elements: key -> phys_tag
    std::map<std::string, int> bnd_face_tag;  // face_key -> phys_tag
    for (auto& e : elems)
        if (is_bnd(e.type))
            bnd_face_tag[face_key(e.nodes)] = e.phys_tag;

    int n_cells = static_cast<int>(raw_cells.size());
    mesh.cells.resize(n_cells);
    mesh.n_cells = n_cells;

    // Compute cell geometry
    for (int ci = 0; ci < n_cells; ++ci) {
        auto& rc = raw_cells[ci];
        std::vector<Vec3> pts;
        pts.reserve(rc.nodes.size());
        for (int ni : rc.nodes) pts.push_back(raw_nodes[ni]);
        compute_cell_geometry(pts, mesh.cells[ci].volume, mesh.cells[ci].center);
        mesh.cells[ci].nodes = rc.nodes;
    }

    // -----------------------------------------------------------------------
    // Build faces by detecting shared faces between cells
    // face_key -> face index (for deduplication)
    // -----------------------------------------------------------------------
    std::map<std::string, int> face_map;  // canonical key -> face index

    for (int ci = 0; ci < n_cells; ++ci) {
        auto& rc = raw_cells[ci];
        auto cell_faces = cell_face_nodes(rc.type, rc.nodes);
        for (auto& fn : cell_faces) {
            std::string key = face_key(fn);
            auto it = face_map.find(key);
            if (it == face_map.end()) {
                // New face — ci is the owner
                int fi = static_cast<int>(mesh.faces.size());
                face_map[key] = fi;
                Face face;
                face.owner = ci;
                face.neighbour = -1;
                face.nodes = fn;
                // Check if boundary element tagged it
                auto bit = bnd_face_tag.find(key);
                if (bit != bnd_face_tag.end()) {
                    int tag = bit->second;
                    face.boundary_tag = phys_names.count(tag)
                                        ? phys_names[tag]
                                        : ("patch_" + std::to_string(tag));
                }
                mesh.faces.push_back(std::move(face));
                mesh.cells[ci].faces.push_back(fi);
            } else {
                // Shared face — ci is the neighbour
                int fi = it->second;
                mesh.faces[fi].neighbour = ci;
                mesh.cells[ci].faces.push_back(fi);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Compute face geometry (area, normal, center)
    // -----------------------------------------------------------------------
    for (auto& face : mesh.faces) {
        std::vector<Vec3> pts;
        pts.reserve(face.nodes.size());
        for (int ni : face.nodes) pts.push_back(raw_nodes[ni]);
        Vec3 owner_center = mesh.cells[face.owner].center;
        compute_face_geometry(pts, owner_center,
                              face.area, face.normal, face.center);
    }

    // -----------------------------------------------------------------------
    // Partition faces: internal first, then boundary
    // -----------------------------------------------------------------------
    std::vector<Face> internal_faces, boundary_faces;
    for (auto& face : mesh.faces) {
        if (face.neighbour >= 0) internal_faces.push_back(face);
        else                     boundary_faces.push_back(face);
    }

    mesh.faces.clear();
    mesh.faces.reserve(internal_faces.size() + boundary_faces.size());
    for (auto& f : internal_faces) mesh.faces.push_back(std::move(f));
    for (auto& f : boundary_faces) mesh.faces.push_back(std::move(f));

    mesh.n_internal_faces  = static_cast<int>(internal_faces.size());
    mesh.n_boundary_faces  = static_cast<int>(boundary_faces.size());
    mesh.n_faces           = static_cast<int>(mesh.faces.size());

    // Rebuild cell face lists to match new face ordering
    // Reset and rebuild using face_map with updated indices
    for (auto& cell : mesh.cells) cell.faces.clear();

    // Rebuild face_map with new indices
    std::map<std::string, int> new_face_map;
    for (int fi = 0; fi < mesh.n_faces; ++fi) {
        auto& face = mesh.faces[fi];
        std::string key = face_key(face.nodes);
        new_face_map[key] = fi;
        mesh.cells[face.owner].faces.push_back(fi);
        if (face.neighbour >= 0)
            mesh.cells[face.neighbour].faces.push_back(fi);
    }

    // -----------------------------------------------------------------------
    // Build boundary_patches map
    // -----------------------------------------------------------------------
    for (int fi = mesh.n_internal_faces; fi < mesh.n_faces; ++fi) {
        auto& face = mesh.faces[fi];
        if (!face.boundary_tag.empty())
            mesh.boundary_patches[face.boundary_tag].push_back(fi);
        else
            mesh.boundary_patches["default"].push_back(fi);
    }

    // -----------------------------------------------------------------------
    // Build cell_zones from physical tags on volume elements
    // -----------------------------------------------------------------------
    for (int ci = 0; ci < n_cells; ++ci) {
        int tag = raw_cells[ci].phys_tag;
        if (tag != 0) {
            std::string zone_name = phys_names.count(tag)
                                    ? phys_names[tag]
                                    : ("zone_" + std::to_string(tag));
            mesh.cell_zones[zone_name].push_back(ci);
        }
    }

    mesh.build_boundary_face_cache();
    return mesh;
}

} // namespace twofluid
