// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef UTIL_HDF5_IO_HDF5_IO_H_
#define UTIL_HDF5_IO_HDF5_IO_H_

#include <Eigen/Dense>
#include <string>
#include <type_traits>
#include <vector>

#include "H5Cpp.h"

namespace hdf5_io {

// Eigen defaults to column-major ordering, but we prefer row-major for two
// reasons:
// - better compatibility with Python bindings: numpy defaults to row-major
// - linear indexing in C++: we have some code that expects to be able to
//   iterate linearly through the elements of the matrix in row-major order
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void WriteH5Dataset(const RowMatrixXd& m, const std::string& name,
                    H5::H5File& file);

// VectorXd and VectorXi (column vectors, i.e. Eigen matrices with 1 column)
template <typename Vector>
  requires(Vector::ColsAtCompileTime == 1) ||
          std::is_same_v<Vector, std::vector<double>> ||
          std::is_same_v<Vector, std::vector<int>>
void WriteH5Dataset(const Vector& v, const std::string& name,
                    H5::H5File& file) {
  const hsize_t size = v.size();
  H5::DataSpace dataspace(/*rank=*/1, /*dims=*/&size);

  H5::PredType type = H5::PredType::NATIVE_DOUBLE;
  if constexpr (std::is_integral_v<typename Vector::value_type>) {
    type = H5::PredType::NATIVE_INT;
  }

  H5::DataSet dataset = file.createDataSet(name, type, dataspace);
  if constexpr (std::is_convertible_v<Vector, Eigen::VectorXd> ||
                std::is_convertible_v<Vector, Eigen::VectorXi>) {
    dataset.write(v.eval().data(), type);
  } else {
    dataset.write(v.data(), type);
  }
}

extern template void WriteH5Dataset(const Eigen::VectorXd& v,
                                    const std::string& name, H5::H5File& file);
extern template void WriteH5Dataset(const Eigen::VectorXi& v,
                                    const std::string& name, H5::H5File& file);

void WriteH5Dataset(const std::vector<std::string>& vs, const std::string& name,
                    H5::H5File& file);

void WriteH5Dataset(const std::string& str, const std::string& name,
                    H5::H5File& file);

template <typename Scalar>
  requires std::is_scalar_v<Scalar>
void WriteH5Dataset(Scalar v, const std::string& name, H5::H5File& file) {
  H5::DataSpace dataspace(H5S_SCALAR);

  H5::PredType type = H5::PredType::NATIVE_DOUBLE;
  if constexpr (std::is_same_v<Scalar, int>) {
    type = H5::PredType::NATIVE_INT;
  } else if (std::is_same_v<Scalar, bool>) {
    type = H5::PredType::NATIVE_HBOOL;
  }

  H5::DataSet dataset = file.createDataSet(name, type, dataspace);
  dataset.write(&v, type);
}

void ReadH5Dataset(RowMatrixXd& m, const std::string& dataset,
                   H5::H5File& file);

// VectorXd and VectorXi (column vectors, i.e. Eigen matrices with 1 column)
template <typename Vector>
  requires std::is_same_v<Vector, Eigen::VectorXd> ||
           std::is_same_v<Vector, Eigen::VectorXi> ||
           std::is_same_v<Vector, std::vector<double>> ||
           std::is_same_v<Vector, std::vector<int>>
void ReadH5Dataset(Vector& v, const std::string& dataset, H5::H5File& file) {
  H5::DataSet ds = file.openDataSet(dataset);

  H5::DataSpace space = ds.getSpace();
  hsize_t size;
  space.getSimpleExtentDims(&size);

  H5::PredType type = H5::PredType::NATIVE_DOUBLE;
  if constexpr (std::is_same_v<Vector, Eigen::VectorXi> ||
                std::is_same_v<Vector, std::vector<int>>) {
    type = H5::PredType::NATIVE_INT;
  }

  v.resize(static_cast<Eigen::Index>(size));
  ds.read(v.data(), type);
}

void ReadH5Dataset(std::vector<std::string>& vs, const std::string& dataset,
                   H5::H5File& file);

void ReadH5Dataset(std::string& str, const std::string& dataset,
                   H5::H5File& file);

template <typename Scalar>
  requires std::is_same_v<Scalar, double> || std::is_same_v<Scalar, int> ||
           std::is_same_v<Scalar, bool>
void ReadH5Dataset(Scalar& v, const std::string& dataset, H5::H5File& file) {
  H5::DataSet ds = file.openDataSet(dataset);

  H5::PredType type = H5::PredType::NATIVE_DOUBLE;
  if constexpr (std::is_same_v<Scalar, int>) {
    type = H5::PredType::NATIVE_INT;
  } else if (std::is_same_v<Scalar, bool>) {
    type = H5::PredType::NATIVE_HBOOL;
  }

  ds.read(&v, type);
}

extern template void ReadH5Dataset(double& v, const std::string& dataset,
                                   H5::H5File& file);
extern template void ReadH5Dataset(int& v, const std::string& dataset,
                                   H5::H5File& file);
extern template void ReadH5Dataset(bool& v, const std::string& dataset,
                                   H5::H5File& file);

// Determine the rank of a given HDF5 dataset,
// i.e., the number of array dimensions
int GetRank(const H5::DataSet& dataset);

// Get the extent, i.e., the size along each dimension,
// of a given HDF5 dataset
std::vector<hsize_t> GetExtent(const H5::DataSet& dataset);

// Read an D dimensional dataset into a linear array
template <typename Vector>
  requires std::is_same_v<Vector, Eigen::VectorXd> ||
           std::is_same_v<Vector, Eigen::VectorXi> ||
           std::is_same_v<Vector, std::vector<double>> ||
           std::is_same_v<Vector, std::vector<int>>
void ReadH5Dataset(Vector& m_vector, std::vector<hsize_t>& m_dims,
                   const std::string& name, H5::H5File& file) {
  H5::DataSet dataset = file.openDataSet(name);
  m_dims = GetExtent(dataset);

  // The underlying array size is the product of all dimensions.
  hsize_t total_size = 1;
  for (auto dim : m_dims) {
    total_size *= dim;
  }

  // All dimensions are flattened into a linear index
  m_vector.resize(static_cast<Eigen::Index>(total_size));

  H5::PredType type = H5::PredType::NATIVE_DOUBLE;
  if constexpr (std::is_integral_v<typename Vector::value_type>) {
    type = H5::PredType::NATIVE_INT;
  }

  dataset.read(m_vector.data(), type);
}  // ReadH5Dataset

// Write a multi-dimensional tensor from contiguous, row major memory to an H5
// dataset as a D dimensional tensor.
template <typename Vector>
  requires(Vector::ColsAtCompileTime == 1) ||
          std::is_same_v<Vector, std::vector<double>> ||
          std::is_same_v<Vector, std::vector<int>>
void WriteH5Dataset(const Vector& vector, const std::vector<hsize_t>& dims,
                    const std::string& name, H5::H5File& h5file_output) {
  H5::DataSpace dataspace(/*rank=*/dims.size(), /*dims=*/dims.data());

  H5::PredType type = H5::PredType::NATIVE_DOUBLE;
  if constexpr (std::is_integral_v<typename Vector::value_type>) {
    type = H5::PredType::NATIVE_INT;
  }
  H5::DataSet dataset = h5file_output.createDataSet(name, type, dataspace);

  dataset.write(vector.data(), type);
}

}  // namespace hdf5_io
#endif  // UTIL_HDF5_IO_HDF5_IO_H_
