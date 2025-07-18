// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <iostream>

#include "util/file_io/file_io.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

using file_io::ReadFile;
using vmecpp::FourierGeometry;
using vmecpp::RadialPartitioning;
using vmecpp::Sizes;
using vmecpp::VmecINDATA;

namespace vmecpp {

TEST(DebugAsymmetricTest, CompareArraySizes) {
  // Load the CTH-like stellarator configuration
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  // Compare symmetric vs asymmetric sizes
  VmecINDATA indata_sym = *indata;
  VmecINDATA indata_asym = *indata;

  indata_sym.lasym = false;
  indata_asym.lasym = true;

  Sizes sizes_sym(indata_sym);
  Sizes sizes_asym(indata_asym);

  std::cout << "=== SYMMETRIC vs ASYMMETRIC SIZE COMPARISON ===" << std::endl;
  std::cout << "Symmetric  (lasym=false):" << std::endl;
  std::cout << "  nfp=" << sizes_sym.nfp << ", mpol=" << sizes_sym.mpol
            << ", ntor=" << sizes_sym.ntor << std::endl;
  std::cout << "  lthreed=" << sizes_sym.lthreed
            << ", num_basis=" << sizes_sym.num_basis << std::endl;
  std::cout << "  nThetaEff=" << sizes_sym.nThetaEff
            << ", nZnT=" << sizes_sym.nZnT << std::endl;
  std::cout << "  mnmax=" << sizes_sym.mnmax << ", mnsize=" << sizes_sym.mnsize
            << std::endl;

  std::cout << "Asymmetric (lasym=true):" << std::endl;
  std::cout << "  nfp=" << sizes_asym.nfp << ", mpol=" << sizes_asym.mpol
            << ", ntor=" << sizes_asym.ntor << std::endl;
  std::cout << "  lthreed=" << sizes_asym.lthreed
            << ", num_basis=" << sizes_asym.num_basis << std::endl;
  std::cout << "  nThetaEff=" << sizes_asym.nThetaEff
            << ", nZnT=" << sizes_asym.nZnT << std::endl;
  std::cout << "  mnmax=" << sizes_asym.mnmax
            << ", mnsize=" << sizes_asym.mnsize << std::endl;

  // Create radial partitioning
  RadialPartitioning radial_partitioning;
  radial_partitioning.adjustRadialPartitioning(1, 0, 25, false, false);

  // Try to create FourierGeometry objects
  std::cout << "Creating FourierGeometry objects..." << std::endl;

  try {
    FourierGeometry geom_sym(&sizes_sym, &radial_partitioning, 25);
    std::cout << "Symmetric FourierGeometry created successfully" << std::endl;
    std::cout << "  rmncc.size()=" << geom_sym.rmncc.size() << std::endl;
    std::cout << "  rmnsc.size()=" << geom_sym.rmnsc.size() << std::endl;
    std::cout << "  zmncc.size()=" << geom_sym.zmncc.size() << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Symmetric FourierGeometry failed: " << e.what() << std::endl;
  }

  try {
    FourierGeometry geom_asym(&sizes_asym, &radial_partitioning, 25);
    std::cout << "Asymmetric FourierGeometry created successfully" << std::endl;
    std::cout << "  rmncc.size()=" << geom_asym.rmncc.size() << std::endl;
    std::cout << "  rmnsc.size()=" << geom_asym.rmnsc.size() << std::endl;
    std::cout << "  zmncc.size()=" << geom_asym.zmncc.size() << std::endl;
  } catch (const std::exception& e) {
    std::cout << "Asymmetric FourierGeometry failed: " << e.what() << std::endl;
  }
}

}  // namespace vmecpp
