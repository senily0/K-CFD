/// OpenFOAM polyMesh ASCII reader
/// Reads constant/polyMesh/{points,faces,owner,neighbour,boundary}
/// and builds an FVMesh supporting arbitrary polyhedral cells.
#include "twofluid/polymesh_reader.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace twofluid {

// ---------------------------------------------------------------------------
// Geometry helpers (identical to mesh_reader.cpp)
// ---------------------------------------------------------------------------
static Vec3 pm_cross3(const Vec3& a, const Vec3& b) {
    return Vec3(a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]);
}

/// Face area, outward normal, and center via Newell's method.
/// Normal is oriented away from owner_center.
static void pm_face_geometry(const std::vector<Vec3>& pts,
                              const Vec3& owner_center,
                              double& area, Vec3& normal, Vec3& center)
{
    int n = static_cast<int>(pts.size());
    center = Vec3::Zero();
    for (const auto& p : pts) center += p;
    center /= static_cast<double>(n);

    // Newell's method: sum of cross products of consecutive edge pairs about center
    Vec3 n_sum = Vec3::Zero();
    for (int i = 0; i < n; ++i) {
        Vec3 a = pts[i]          - center;
        Vec3 b = pts[(i+1) % n] - center;
        n_sum += pm_cross3(a, b);
    }
    area = 0.5 * n_sum.norm();
    if (area < 1e-300) {
        normal = Vec3(0.0, 0.0, 1.0);
        return;
    }
    normal = n_sum / n_sum.norm();
    if (normal.dot(center - owner_center) < 0.0)
        normal = -normal;
}

/// Cell center (area-weighted average of face centers) and volume
/// via divergence theorem: V = (1/3) * sum_f (x_f . n_f * A_f).
static void pm_cell_geometry(const std::vector<Vec3>& face_centers,
                              const std::vector<Vec3>& face_normals,
                              const std::vector<double>& face_areas,
                              double& volume, Vec3& center)
{
    int nf = static_cast<int>(face_centers.size());
    double total_area = 0.0;
    center = Vec3::Zero();
    for (int i = 0; i < nf; ++i) {
        center     += face_areas[i] * face_centers[i];
        total_area += face_areas[i];
    }
    if (total_area > 1e-300)
        center /= total_area;

    // Divergence theorem: V = (1/3) sum_f (f_center . f_normal * f_area)
    volume = 0.0;
    for (int i = 0; i < nf; ++i)
        volume += face_centers[i].dot(face_normals[i]) * face_areas[i];
    volume = std::abs(volume) / 3.0;
}

// ---------------------------------------------------------------------------
// OpenFOAM file parsing helpers
// ---------------------------------------------------------------------------

/// Skip the FoamFile header block and any blank/comment lines.
/// Leaves the stream positioned at the integer count line.
static void skip_foam_header(std::istream& in) {
    std::string line;
    bool in_foam_block = false;
    int brace_depth = 0;
    while (std::getline(in, line)) {
        // Strip inline comments
        auto pos = line.find("//");
        if (pos != std::string::npos) line = line.substr(0, pos);

        if (!in_foam_block) {
            if (line.find("FoamFile") != std::string::npos) {
                in_foam_block = true;
                brace_depth = 0;
            }
            // If we hit a bare integer on a line, we've passed the header —
            // but we haven't consumed it yet, so just continue scanning.
        }

        if (in_foam_block) {
            for (char c : line) {
                if (c == '{') ++brace_depth;
                if (c == '}') --brace_depth;
            }
            if (brace_depth <= 0 && in_foam_block && line.find('}') != std::string::npos) {
                // FoamFile block closed; next non-blank non-comment line is the count
                return;
            }
        }
    }
}

/// After skip_foam_header(), read the integer count line and opening '('.
/// Returns the count N.
static int read_count_and_open(std::istream& in) {
    std::string line;
    int n = -1;
    while (std::getline(in, line)) {
        // Strip comments
        auto pos = line.find("//");
        if (pos != std::string::npos) line = line.substr(0, pos);
        std::istringstream ss(line);
        if (ss >> n && n >= 0) {
            // Now consume the opening '('
            while (std::getline(in, line)) {
                auto p2 = line.find("//");
                if (p2 != std::string::npos) line = line.substr(0, p2);
                if (line.find('(') != std::string::npos) return n;
            }
        }
    }
    throw std::runtime_error("read_openfoam_polymesh: could not read count/opening paren");
}

/// Parse the points file. Returns vector of Vec3.
static std::vector<Vec3> parse_points(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("read_openfoam_polymesh: cannot open points file: " + path);

    skip_foam_header(f);
    int n = read_count_and_open(f);

    std::vector<Vec3> pts;
    pts.reserve(n);

    std::string line;
    while (std::getline(f, line) && static_cast<int>(pts.size()) < n) {
        // Strip comments
        auto pos = line.find("//");
        if (pos != std::string::npos) line = line.substr(0, pos);
        // Each line: (x y z)
        auto lp = line.find('(');
        auto rp = line.find(')');
        if (lp == std::string::npos || rp == std::string::npos) continue;
        std::string inner = line.substr(lp + 1, rp - lp - 1);
        std::istringstream ss(inner);
        double x, y, z;
        if (ss >> x >> y >> z)
            pts.push_back(Vec3(x, y, z));
    }

    if (static_cast<int>(pts.size()) != n)
        throw std::runtime_error("read_openfoam_polymesh: expected " + std::to_string(n)
                                 + " points, got " + std::to_string(pts.size()));
    return pts;
}

/// Parse the faces file. Returns vector of face node lists (0-based).
static std::vector<std::vector<int>> parse_faces(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("read_openfoam_polymesh: cannot open faces file: " + path);

    skip_foam_header(f);
    int n = read_count_and_open(f);

    std::vector<std::vector<int>> faces;
    faces.reserve(n);

    std::string line;
    while (std::getline(f, line) && static_cast<int>(faces.size()) < n) {
        auto pos = line.find("//");
        if (pos != std::string::npos) line = line.substr(0, pos);
        // Format: N(n0 n1 n2 ... nN-1)
        auto lp = line.find('(');
        auto rp = line.rfind(')');
        if (lp == std::string::npos || rp == std::string::npos) continue;

        // The count before '('
        std::string count_str = line.substr(0, lp);
        std::istringstream cs(count_str);
        int fn;
        if (!(cs >> fn)) continue;

        std::string inner = line.substr(lp + 1, rp - lp - 1);
        std::istringstream ss(inner);
        std::vector<int> nodes;
        nodes.reserve(fn);
        int nid;
        while (ss >> nid) nodes.push_back(nid);

        if (static_cast<int>(nodes.size()) == fn)
            faces.push_back(std::move(nodes));
    }

    if (static_cast<int>(faces.size()) != n)
        throw std::runtime_error("read_openfoam_polymesh: expected " + std::to_string(n)
                                 + " faces, got " + std::to_string(faces.size()));
    return faces;
}

/// Parse owner or neighbour file. Returns vector of cell indices.
static std::vector<int> parse_label_list(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("read_openfoam_polymesh: cannot open file: " + path);

    skip_foam_header(f);
    int n = read_count_and_open(f);

    std::vector<int> labels;
    labels.reserve(n);

    std::string line;
    while (std::getline(f, line) && static_cast<int>(labels.size()) < n) {
        auto pos = line.find("//");
        if (pos != std::string::npos) line = line.substr(0, pos);
        std::istringstream ss(line);
        int v;
        while (ss >> v && static_cast<int>(labels.size()) < n)
            labels.push_back(v);
    }

    if (static_cast<int>(labels.size()) != n)
        throw std::runtime_error("read_openfoam_polymesh: expected " + std::to_string(n)
                                 + " labels, got " + std::to_string(labels.size()));
    return labels;
}

struct BoundaryPatch {
    std::string name;
    int n_faces;
    int start_face;
};

/// Parse the boundary file. Returns list of BoundaryPatch.
static std::vector<BoundaryPatch> parse_boundary(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("read_openfoam_polymesh: cannot open boundary file: " + path);

    skip_foam_header(f);

    // Read patch count
    std::string line;
    int n_patches = 0;
    while (std::getline(f, line)) {
        auto pos = line.find("//");
        if (pos != std::string::npos) line = line.substr(0, pos);
        std::istringstream ss(line);
        if (ss >> n_patches && n_patches >= 0) {
            // consume opening '('
            while (std::getline(f, line)) {
                if (line.find('(') != std::string::npos) break;
            }
            break;
        }
    }

    std::vector<BoundaryPatch> patches;
    patches.reserve(n_patches);

    for (int i = 0; i < n_patches; ++i) {
        BoundaryPatch bp;
        bp.n_faces    = 0;
        bp.start_face = 0;

        // Read patch name (first non-blank line)
        while (std::getline(f, line)) {
            auto pos = line.find("//");
            if (pos != std::string::npos) line = line.substr(0, pos);
            std::istringstream ss(line);
            std::string token;
            if (ss >> token && token != "{" && token != "}") {
                bp.name = token;
                break;
            }
        }
        // Read '{' then key-value pairs until '}'
        bool in_block = false;
        while (std::getline(f, line)) {
            auto pos = line.find("//");
            if (pos != std::string::npos) line = line.substr(0, pos);
            if (line.find('{') != std::string::npos) { in_block = true; continue; }
            if (line.find('}') != std::string::npos) break;
            if (!in_block) continue;
            std::istringstream ss(line);
            std::string key;
            if (!(ss >> key)) continue;
            if (key == "nFaces")    ss >> bp.n_faces;
            if (key == "startFace") ss >> bp.start_face;
        }
        patches.push_back(bp);
    }

    return patches;
}

// ---------------------------------------------------------------------------
// Main reader
// ---------------------------------------------------------------------------
FVMesh read_openfoam_polymesh(const std::string& case_dir) {
    std::string pm_dir = case_dir + "/constant/polyMesh";

    // 1. Parse all five files
    auto raw_points     = parse_points    (pm_dir + "/points");
    auto raw_faces      = parse_faces     (pm_dir + "/faces");
    auto owner_list     = parse_label_list(pm_dir + "/owner");
    auto boundary_defs  = parse_boundary  (pm_dir + "/boundary");

    // neighbour is optional (empty for pure-boundary meshes, but normally present)
    std::vector<int> neighbour_list;
    {
        std::string nb_path = pm_dir + "/neighbour";
        std::ifstream check(nb_path);
        if (check.is_open()) {
            check.close();
            neighbour_list = parse_label_list(nb_path);
        }
    }

    int n_faces_total = static_cast<int>(raw_faces.size());
    int n_points      = static_cast<int>(raw_points.size());

    if (static_cast<int>(owner_list.size()) != n_faces_total)
        throw std::runtime_error("read_openfoam_polymesh: owner list size mismatch");

    // Determine n_cells from max owner index
    int n_cells = 0;
    for (int o : owner_list)
        if (o + 1 > n_cells) n_cells = o + 1;
    for (int nb : neighbour_list)
        if (nb + 1 > n_cells) n_cells = nb + 1;

    // Determine mesh dimension: 2D if all z == 0
    bool has_z = false;
    for (const auto& p : raw_points)
        if (std::abs(p[2]) > 1e-12) { has_z = true; break; }
    int ndim = has_z ? 3 : 2;

    FVMesh mesh(ndim);

    // 2. Copy nodes
    mesh.nodes.resize(n_points, 3);
    for (int i = 0; i < n_points; ++i)
        mesh.nodes.row(i) = raw_points[i].transpose();

    // 3. Build cells structure
    mesh.cells.resize(n_cells);
    mesh.n_cells = n_cells;

    // 4. Build boundary patch face-range map (face index -> patch name)
    // Internal faces have no neighbour entry, boundary faces start at n_internal_faces.
    int n_internal = static_cast<int>(neighbour_list.size());
    // boundary face indices: [n_internal, n_faces_total)

    // Map face index -> boundary patch name
    std::vector<std::string> face_patch(n_faces_total);  // empty = internal
    for (auto& bp : boundary_defs) {
        for (int fi = bp.start_face; fi < bp.start_face + bp.n_faces; ++fi) {
            if (fi < n_faces_total)
                face_patch[fi] = bp.name;
        }
    }

    // 5. Compute temporary face geometries (need owner center first — bootstrap)
    //    Use node average as a first-pass cell center estimate.
    //    We'll recompute cell centers properly after.

    // First pass: assign faces to cells, compute face geometry with
    // a rough owner center (centroid of all nodes touching the cell).
    // Collect per-cell face centers, normals, areas for cell geometry.

    mesh.faces.resize(n_faces_total);
    mesh.n_faces = n_faces_total;

    // Temporary cell node accumulator for bootstrap center estimate
    std::vector<Vec3>   cell_center_sum(n_cells, Vec3::Zero());
    std::vector<int>    cell_node_count(n_cells, 0);

    for (int fi = 0; fi < n_faces_total; ++fi) {
        const auto& fn = raw_faces[fi];
        int owner = owner_list[fi];
        for (int ni : fn) {
            cell_center_sum[owner] += raw_points[ni];
            cell_node_count[owner]++;
        }
        if (fi < n_internal) {
            int nb = neighbour_list[fi];
            for (int ni : fn) {
                cell_center_sum[nb] += raw_points[ni];
                cell_node_count[nb]++;
            }
        }
    }
    std::vector<Vec3> cell_center_est(n_cells);
    for (int ci = 0; ci < n_cells; ++ci) {
        if (cell_node_count[ci] > 0)
            cell_center_est[ci] = cell_center_sum[ci] / static_cast<double>(cell_node_count[ci]);
        else
            cell_center_est[ci] = Vec3::Zero();
    }

    // 6. Compute face geometry using bootstrap owner center
    for (int fi = 0; fi < n_faces_total; ++fi) {
        Face& face = mesh.faces[fi];
        face.owner     = owner_list[fi];
        face.neighbour = (fi < n_internal) ? neighbour_list[fi] : -1;
        face.nodes     = raw_faces[fi];
        face.boundary_tag = face_patch[fi];

        std::vector<Vec3> pts;
        pts.reserve(face.nodes.size());
        for (int ni : face.nodes) pts.push_back(raw_points[ni]);

        Vec3 owner_est = cell_center_est[face.owner];
        pm_face_geometry(pts, owner_est,
                         face.area, face.normal, face.center);
    }

    // 7. Assign faces to cells
    for (int fi = 0; fi < n_faces_total; ++fi) {
        mesh.cells[mesh.faces[fi].owner].faces.push_back(fi);
        if (mesh.faces[fi].neighbour >= 0)
            mesh.cells[mesh.faces[fi].neighbour].faces.push_back(fi);
    }

    // 8. Compute cell geometry via divergence theorem
    for (int ci = 0; ci < n_cells; ++ci) {
        Cell& cell = mesh.cells[ci];
        std::vector<Vec3>   fc_list, fn_list;
        std::vector<double> fa_list;
        for (int fi : cell.faces) {
            const Face& face = mesh.faces[fi];
            fc_list.push_back(face.center);
            // Orient normal outward from this cell
            Vec3 n_out = (face.owner == ci) ? face.normal : -face.normal;
            fn_list.push_back(n_out);
            fa_list.push_back(face.area);
        }
        pm_cell_geometry(fc_list, fn_list, fa_list, cell.volume, cell.center);
    }

    // 9. Partition: internal faces first, then boundary (OpenFOAM already orders this way)
    //    OpenFOAM convention: faces [0, n_internal) are internal,
    //    [n_internal, n_faces_total) are boundary. Already correct.
    mesh.n_internal_faces = n_internal;
    mesh.n_boundary_faces = n_faces_total - n_internal;

    // 10. Build boundary_patches map
    for (int fi = n_internal; fi < n_faces_total; ++fi) {
        const std::string& tag = mesh.faces[fi].boundary_tag;
        if (!tag.empty())
            mesh.boundary_patches[tag].push_back(fi);
        else
            mesh.boundary_patches["default"].push_back(fi);
    }

    mesh.build_boundary_face_cache();
    return mesh;
}

} // namespace twofluid
