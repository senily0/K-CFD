#include "twofluid/vtk_writer.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace twofluid {

void write_vtu(
    const std::string& filename,
    const FVMesh& mesh,
    const std::unordered_map<std::string, Eigen::VectorXd>& cell_scalar_data,
    const std::unordered_map<std::string, Eigen::MatrixXd>& cell_vector_data)
{
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    ofs << std::setprecision(15);

    int n_points = static_cast<int>(mesh.nodes.rows());
    int n_cells = mesh.n_cells;

    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    ofs << "  <UnstructuredGrid>\n";
    ofs << "    <Piece NumberOfPoints=\"" << n_points
        << "\" NumberOfCells=\"" << n_cells << "\">\n";

    // Points
    ofs << "      <Points>\n";
    ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    ofs << "          ";
    for (int i = 0; i < n_points; ++i) {
        double x = mesh.nodes(i, 0);
        double y = mesh.nodes(i, 1);
        double z = (mesh.nodes.cols() >= 3) ? mesh.nodes(i, 2) : 0.0;
        ofs << x << " " << y << " " << z;
        if (i + 1 < n_points) ofs << " ";
    }
    ofs << "\n";
    ofs << "        </DataArray>\n";
    ofs << "      </Points>\n";

    // Cells
    ofs << "      <Cells>\n";

    // Connectivity
    ofs << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    ofs << "          ";
    for (int i = 0; i < n_cells; ++i) {
        const auto& cell = mesh.cells[i];
        for (size_t j = 0; j < cell.nodes.size(); ++j) {
            if (i > 0 || j > 0) ofs << " ";
            ofs << cell.nodes[j];
        }
    }
    ofs << "\n";
    ofs << "        </DataArray>\n";

    // Offsets
    ofs << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    ofs << "          ";
    int offset = 0;
    for (int i = 0; i < n_cells; ++i) {
        offset += static_cast<int>(mesh.cells[i].nodes.size());
        if (i > 0) ofs << " ";
        ofs << offset;
    }
    ofs << "\n";
    ofs << "        </DataArray>\n";

    // Types
    ofs << "        <DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">\n";
    ofs << "          ";
    for (int i = 0; i < n_cells; ++i) {
        if (i > 0) ofs << " ";
        int nn = static_cast<int>(mesh.cells[i].nodes.size());
        int vtk_type;
        switch (nn) {
            case 3:  vtk_type = 5;  break;  // VTK_TRIANGLE
            case 4:  vtk_type = 9;  break;  // VTK_QUAD
            case 8:  vtk_type = 12; break;  // VTK_HEXAHEDRON
            default: vtk_type = 7;  break;  // VTK_POLYGON
        }
        ofs << vtk_type;
    }
    ofs << "\n";
    ofs << "        </DataArray>\n";
    ofs << "      </Cells>\n";

    // CellData
    if (!cell_scalar_data.empty() || !cell_vector_data.empty()) {
        ofs << "      <CellData>\n";

        for (const auto& [name, data] : cell_scalar_data) {
            if (data.size() != n_cells) continue;
            ofs << "        <DataArray type=\"Float64\" Name=\"" << name
                << "\" format=\"ascii\">\n";
            ofs << "          ";
            for (int i = 0; i < n_cells; ++i) {
                if (i > 0) ofs << " ";
                ofs << data(i);
            }
            ofs << "\n";
            ofs << "        </DataArray>\n";
        }

        for (const auto& [name, data] : cell_vector_data) {
            if (data.rows() != n_cells) continue;
            int ncomp = static_cast<int>(data.cols());
            ofs << "        <DataArray type=\"Float64\" Name=\"" << name
                << "\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            ofs << "          ";
            for (int i = 0; i < n_cells; ++i) {
                if (i > 0) ofs << " ";
                for (int c = 0; c < 3; ++c) {
                    if (c > 0) ofs << " ";
                    ofs << ((c < ncomp) ? data(i, c) : 0.0);
                }
            }
            ofs << "\n";
            ofs << "        </DataArray>\n";
        }

        ofs << "      </CellData>\n";
    }

    ofs << "    </Piece>\n";
    ofs << "  </UnstructuredGrid>\n";
    ofs << "</VTKFile>\n";
}

// ---------------------------------------------------------------------------
// Binary (appended raw) VTU writer
// ---------------------------------------------------------------------------

// Helper: append raw bytes of a typed array to a byte buffer
static void append_uint32(std::vector<uint8_t>& buf, uint32_t v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), p, p + 4);
}
static void append_bytes(std::vector<uint8_t>& buf, const void* data, size_t n_bytes) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    buf.insert(buf.end(), p, p + n_bytes);
}

void write_vtu_binary(
    const std::string& filename,
    const FVMesh& mesh,
    const std::unordered_map<std::string, Eigen::VectorXd>& cell_scalar_data,
    const std::unordered_map<std::string, Eigen::MatrixXd>& cell_vector_data)
{
    int n_points = static_cast<int>(mesh.nodes.rows());
    int n_cells  = mesh.n_cells;

    // ---- Build appended data buffer and track offsets ----
    std::vector<uint8_t> appended;

    // Reserve slot for offset tracking: each block = 4-byte header + data
    // We record byte offsets *before* writing each block.
    std::vector<uint32_t> offsets; // byte offsets into appended buffer

    // --- Points (Float64, 3 components) ---
    offsets.push_back(static_cast<uint32_t>(appended.size()));
    {
        std::vector<double> pts;
        pts.reserve(n_points * 3);
        for (int i = 0; i < n_points; ++i) {
            pts.push_back(mesh.nodes(i, 0));
            pts.push_back(mesh.nodes(i, 1));
            pts.push_back((mesh.nodes.cols() >= 3) ? mesh.nodes(i, 2) : 0.0);
        }
        uint32_t byte_count = static_cast<uint32_t>(pts.size() * sizeof(double));
        append_uint32(appended, byte_count);
        append_bytes(appended, pts.data(), byte_count);
    }

    // --- Connectivity (Int32) ---
    offsets.push_back(static_cast<uint32_t>(appended.size()));
    {
        std::vector<int32_t> conn;
        for (int i = 0; i < n_cells; ++i)
            for (int n : mesh.cells[i].nodes)
                conn.push_back(static_cast<int32_t>(n));
        uint32_t byte_count = static_cast<uint32_t>(conn.size() * sizeof(int32_t));
        append_uint32(appended, byte_count);
        append_bytes(appended, conn.data(), byte_count);
    }

    // --- Offsets (Int32) ---
    offsets.push_back(static_cast<uint32_t>(appended.size()));
    {
        std::vector<int32_t> off_arr;
        int32_t running = 0;
        for (int i = 0; i < n_cells; ++i) {
            running += static_cast<int32_t>(mesh.cells[i].nodes.size());
            off_arr.push_back(running);
        }
        uint32_t byte_count = static_cast<uint32_t>(off_arr.size() * sizeof(int32_t));
        append_uint32(appended, byte_count);
        append_bytes(appended, off_arr.data(), byte_count);
    }

    // --- Types (Int32) ---
    offsets.push_back(static_cast<uint32_t>(appended.size()));
    {
        std::vector<int32_t> types;
        for (int i = 0; i < n_cells; ++i) {
            int nn = static_cast<int>(mesh.cells[i].nodes.size());
            int32_t vt;
            switch (nn) {
                case 3:  vt = 5;  break;
                case 4:  vt = 9;  break;
                case 8:  vt = 12; break;
                default: vt = 7;  break;
            }
            types.push_back(vt);
        }
        uint32_t byte_count = static_cast<uint32_t>(types.size() * sizeof(int32_t));
        append_uint32(appended, byte_count);
        append_bytes(appended, types.data(), byte_count);
    }

    // --- Scalar CellData ---
    std::vector<std::string> scalar_names_ordered;
    for (const auto& [name, data] : cell_scalar_data)
        if (data.size() == n_cells) scalar_names_ordered.push_back(name);

    for (const auto& name : scalar_names_ordered) {
        const auto& data = cell_scalar_data.at(name);
        offsets.push_back(static_cast<uint32_t>(appended.size()));
        uint32_t byte_count = static_cast<uint32_t>(n_cells * sizeof(double));
        append_uint32(appended, byte_count);
        append_bytes(appended, data.data(), byte_count);
    }

    // --- Vector CellData ---
    std::vector<std::string> vector_names_ordered;
    for (const auto& [name, data] : cell_vector_data)
        if (data.rows() == n_cells) vector_names_ordered.push_back(name);

    for (const auto& name : vector_names_ordered) {
        const auto& data = cell_vector_data.at(name);
        int ncomp = static_cast<int>(data.cols());
        offsets.push_back(static_cast<uint32_t>(appended.size()));
        std::vector<double> vec;
        vec.reserve(n_cells * 3);
        for (int i = 0; i < n_cells; ++i)
            for (int c = 0; c < 3; ++c)
                vec.push_back((c < ncomp) ? data(i, c) : 0.0);
        uint32_t byte_count = static_cast<uint32_t>(vec.size() * sizeof(double));
        append_uint32(appended, byte_count);
        append_bytes(appended, vec.data(), byte_count);
    }

    // ---- Write XML header ----
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open())
        throw std::runtime_error("Cannot open file for writing: " + filename);

    int off_idx = 0;
    ofs << "<?xml version=\"1.0\"?>\n";
    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\""
           " byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
    ofs << "  <UnstructuredGrid>\n";
    ofs << "    <Piece NumberOfPoints=\"" << n_points
        << "\" NumberOfCells=\"" << n_cells << "\">\n";

    ofs << "      <Points>\n";
    ofs << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\""
           " format=\"appended\" offset=\"" << offsets[off_idx++] << "\"/>\n";
    ofs << "      </Points>\n";

    ofs << "      <Cells>\n";
    ofs << "        <DataArray type=\"Int32\" Name=\"connectivity\""
           " format=\"appended\" offset=\"" << offsets[off_idx++] << "\"/>\n";
    ofs << "        <DataArray type=\"Int32\" Name=\"offsets\""
           " format=\"appended\" offset=\"" << offsets[off_idx++] << "\"/>\n";
    ofs << "        <DataArray type=\"Int32\" Name=\"types\""
           " format=\"appended\" offset=\"" << offsets[off_idx++] << "\"/>\n";
    ofs << "      </Cells>\n";

    if (!scalar_names_ordered.empty() || !vector_names_ordered.empty()) {
        ofs << "      <CellData>\n";
        for (const auto& name : scalar_names_ordered) {
            ofs << "        <DataArray type=\"Float64\" Name=\"" << name
                << "\" format=\"appended\" offset=\"" << offsets[off_idx++] << "\"/>\n";
        }
        for (const auto& name : vector_names_ordered) {
            ofs << "        <DataArray type=\"Float64\" Name=\"" << name
                << "\" NumberOfComponents=\"3\""
                << " format=\"appended\" offset=\"" << offsets[off_idx++] << "\"/>\n";
        }
        ofs << "      </CellData>\n";
    }

    ofs << "    </Piece>\n";
    ofs << "  </UnstructuredGrid>\n";
    ofs << "  <AppendedData encoding=\"raw\">\n";
    ofs << "    _";
    ofs.write(reinterpret_cast<const char*>(appended.data()),
              static_cast<std::streamsize>(appended.size()));
    ofs << "\n  </AppendedData>\n";
    ofs << "</VTKFile>\n";
}

} // namespace twofluid
