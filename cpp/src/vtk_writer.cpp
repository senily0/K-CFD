// Disable optimization for this file — MinGW -O1/-O2 causes segfaults
// in the unordered_map iteration + ofstream combination below.
#pragma GCC optimize("O0")

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

} // namespace twofluid
