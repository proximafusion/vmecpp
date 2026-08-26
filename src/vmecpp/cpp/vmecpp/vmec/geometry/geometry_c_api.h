// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_GEOMETRY_GEOMETRY_C_API_H_
#define VMECPP_VMEC_GEOMETRY_GEOMETRY_C_API_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// C ABI names and arrays intentionally follow C conventions.
// NOLINTNEXTLINE(modernize-use-using)
typedef struct vmecpp_geometry_handle vmecpp_geometry_handle;

// Entries are value, d/ds, d/dtheta, d/dzeta, d2/ds2, d2/(ds dtheta),
// d2/(ds dzeta), d2/dtheta2, d2/(dtheta dzeta), and d2/dzeta2.
#define VMECPP_GEOMETRY_JET_SIZE 10
// NOLINTNEXTLINE(modernize-use-using,readability-identifier-naming)
typedef struct vmecpp_geometry_point {
  double r[VMECPP_GEOMETRY_JET_SIZE];
  double z[VMECPP_GEOMETRY_JET_SIZE];
  double lambda[VMECPP_GEOMETRY_JET_SIZE];
  double toroidal_flux[VMECPP_GEOMETRY_JET_SIZE];
  double poloidal_flux[VMECPP_GEOMETRY_JET_SIZE];
} vmecpp_geometry_point;

// NOLINTNEXTLINE(modernize-use-using,readability-identifier-naming)
typedef struct vmecpp_geometry_metadata {
  int nfp;
  double major_radius;
} vmecpp_geometry_metadata;

// Create a geometry handle from a VMEC++ input file. This runs VMEC++ in
// memory and retains the resulting geometry; no wout file is read or written.
// Returns zero on success. On failure, vmecpp_geometry_error() describes the
// error for the calling thread.
int vmecpp_geometry_create(const char* input_path,
                           vmecpp_geometry_handle** output);
void vmecpp_geometry_destroy(vmecpp_geometry_handle* handle);
int vmecpp_geometry_get_metadata(const vmecpp_geometry_handle* handle,
                                 vmecpp_geometry_metadata* output);
int vmecpp_geometry_evaluate(const vmecpp_geometry_handle* handle, double s,
                             double theta, double zeta,
                             vmecpp_geometry_point* output);
const char* vmecpp_geometry_error(void);

#ifdef __cplusplus
}
#endif

#endif  // VMECPP_VMEC_GEOMETRY_GEOMETRY_C_API_H_
