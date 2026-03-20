#include "twofluid/mpi_comm.hpp"
#include <stdexcept>

namespace twofluid {

MPIComm::MPIComm() {
#ifdef USE_MPI
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) {
        MPI_Init(nullptr, nullptr);
        owns_mpi_ = true;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
#else
    rank_ = 0;
    size_ = 1;
#endif
}

MPIComm::~MPIComm() {
#ifdef USE_MPI
    if (owns_mpi_) {
        int flag = 0;
        MPI_Finalized(&flag);
        if (!flag) {
            MPI_Finalize();
        }
    }
#endif
}

void MPIComm::barrier() {
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

double MPIComm::all_reduce_sum(double val) {
#ifdef USE_MPI
    double result = 0.0;
    MPI_Allreduce(&val, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return result;
#else
    return val;
#endif
}

double MPIComm::all_reduce_max(double val) {
#ifdef USE_MPI
    double result = 0.0;
    MPI_Allreduce(&val, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
#else
    return val;
#endif
}

double MPIComm::all_reduce_min(double val) {
#ifdef USE_MPI
    double result = 0.0;
    MPI_Allreduce(&val, &result, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return result;
#else
    return val;
#endif
}

void MPIComm::send_scalar(const Eigen::VectorXd& data,
                          const std::vector<int>& indices,
                          int dest, int tag) {
#ifdef USE_MPI
    std::vector<double> buf(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        buf[i] = data[indices[i]];
    }
    MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE,
             dest, tag, MPI_COMM_WORLD);
#else
    (void)data; (void)indices; (void)dest; (void)tag;
#endif
}

Eigen::VectorXd MPIComm::recv_scalar(int count, int src, int tag) {
    Eigen::VectorXd result(count);
#ifdef USE_MPI
    MPI_Recv(result.data(), count, MPI_DOUBLE,
             src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
    (void)src; (void)tag;
    result.setZero();
#endif
    return result;
}

void MPIComm::send_vector(const Eigen::MatrixXd& data,
                          const std::vector<int>& indices,
                          int dest, int tag) {
#ifdef USE_MPI
    const int ndim = static_cast<int>(data.cols());
    std::vector<double> buf(indices.size() * ndim);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        for (int d = 0; d < ndim; ++d) {
            buf[i * ndim + d] = data(indices[i], d);
        }
    }
    MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE,
             dest, tag, MPI_COMM_WORLD);
#else
    (void)data; (void)indices; (void)dest; (void)tag;
#endif
}

Eigen::MatrixXd MPIComm::recv_vector(int count, int ndim, int src, int tag) {
    Eigen::MatrixXd result(count, ndim);
#ifdef USE_MPI
    std::vector<double> buf(count * ndim);
    MPI_Recv(buf.data(), count * ndim, MPI_DOUBLE,
             src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < count; ++i) {
        for (int d = 0; d < ndim; ++d) {
            result(i, d) = buf[i * ndim + d];
        }
    }
#else
    (void)src; (void)tag;
    result.setZero();
#endif
    return result;
}

} // namespace twofluid
