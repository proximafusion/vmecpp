// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/hdf5_io/hdf5_io.h"

#include <array>
#include <string>
#include <vector>

void hdf5_io::WriteH5Dataset(const hdf5_io::RowMatrixXd& m,
                             const std::string& name, H5::H5File& file) {
  const std::array<hsize_t, 2> dims = {static_cast<hsize_t>(m.rows()),
                                       static_cast<hsize_t>(m.cols())};
  H5::DataSpace dataspace(/*rank=*/2, /*dims=*/dims.data());
  H5::DataSet dataset =
      file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(m.data(), H5::PredType::NATIVE_DOUBLE);
}

template void hdf5_io::WriteH5Dataset(const Eigen::VectorXd& v,
                                      const std::string& name,
                                      H5::H5File& file);

template void hdf5_io::WriteH5Dataset(const Eigen::VectorXi& v,
                                      const std::string& name,
                                      H5::H5File& file);

void hdf5_io::WriteH5Dataset(const std::vector<std::string>& vs,
                             const std::string& name, H5::H5File& file) {
  const hsize_t size = vs.size();
  H5::DataSpace dataspace(/*rank=*/1, /*dims=*/&size);
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSet dataset = file.createDataSet(name, str_type, dataspace);
  dataset.write(vs.data(), str_type);
}

void hdf5_io::WriteH5Dataset(const std::string& str, const std::string& name,
                             H5::H5File& file) {
  H5::DataSpace dataspace(H5S_SCALAR);
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSet dataset = file.createDataSet(name, str_type, dataspace);
  dataset.write(str, str_type);
}

void hdf5_io::ReadH5Dataset(hdf5_io::RowMatrixXd& m, const std::string& dataset,
                            H5::H5File& file) {
  H5::DataSet ds = file.openDataSet(dataset);

  H5::DataSpace space = ds.getSpace();
  std::array<hsize_t, 2> dims;
  space.getSimpleExtentDims(dims.data());

  m.resize(static_cast<Eigen::Index>(dims[0]),
           static_cast<Eigen::Index>(dims[1]));
  ds.read(m.data(), H5::PredType::NATIVE_DOUBLE);
}

void hdf5_io::ReadH5Dataset(std::vector<std::string>& vs,
                            const std::string& dataset, H5::H5File& file) {
  H5::DataSet ds = file.openDataSet(dataset);

  H5::DataSpace space = ds.getSpace();
  hsize_t size;
  space.getSimpleExtentDims(&size);

  vs.resize(size);
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  ds.read(vs.data(), str_type);
}

void hdf5_io::ReadH5Dataset(std::string& str, const std::string& dataset,
                            H5::H5File& file) {
  H5::DataSet ds = file.openDataSet(dataset);
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  ds.read(str, str_type);
}

template void hdf5_io::ReadH5Dataset(double& v, const std::string& dataset,
                                     H5::H5File& file);
template void hdf5_io::ReadH5Dataset(int& v, const std::string& dataset,
                                     H5::H5File& file);
template void hdf5_io::ReadH5Dataset(bool& v, const std::string& dataset,
                                     H5::H5File& file);

int hdf5_io::GetRank(const H5::DataSet& dataset) {
  H5::DataSpace dataspace = dataset.getSpace();
  return dataspace.getSimpleExtentNdims();
}

std::vector<hsize_t> hdf5_io::GetExtent(const H5::DataSet& dataset) {
  H5::DataSpace dataspace = dataset.getSpace();
  int rank = dataspace.getSimpleExtentNdims();
  std::vector<hsize_t> extent(rank);
  dataspace.getSimpleExtentDims(extent.data(), NULL);
  return extent;
}
