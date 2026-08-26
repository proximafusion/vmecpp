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

typedef struct vmecpp_geometry_handle vmecpp_geometry_handle;

// Entries are value, d/ds, d/dtheta, and d/dzeta.
typedef struct vmecpp_geometry_point {
  double r[4];
  double z[4];
  double lambda[4];
  double toroidal_flux[4];
  double poloidal_flux[4];
} vmecpp_geometry_point;

typedef struct vmecpp_geometry_metadata {
  int nfp;
  double major_radius;
} vmecpp_geometry_metadata;

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
