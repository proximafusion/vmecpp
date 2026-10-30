//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for information.
//

#include <stdlib.h>
#include <string.h>

#include <iostream>

#include "fftx_iprdftbat_cpu_decls.h"
#include "fftx_iprdftbat_cpu_public.h"

//  Query the list of sizes available from the library; returns a pointer to an
//  array of size <N+1>, each element is a struct of type fftx::point_t<4>
//  specifying FFT length, number of batches, and read/write stride types. The
//  final entry in the list is a zero entry.

fftx::point_t<4> *fftx_iprdftbat_cpu_QuerySizes() {
  fftx::point_t<4> *wp = (fftx::point_t<4> *)malloc(sizeof(AllSizes4_CPU));
  if (wp != NULL)
    memcpy((void *)wp, (const void *)AllSizes4_CPU, sizeof(AllSizes4_CPU));

  return wp;
}

//  Get a transform tuple -- a set of pointers to the init, destroy, and run
//  functions for a specific size fftx_iprdftbat_ transform.  Using this
//  information the user may call the init function to setup for the transform,
//  then run the transform repeatedly, and finally tear down (using the destroy
//  function).  Returns NULL if requested size is not found

transformTuple_t *fftx_iprdftbat_cpu_Tuple(fftx::point_t<4> req) {
  int indx;
  int numentries = sizeof(AllSizes4_CPU) / sizeof(fftx::point_t<4>) -
                   1;  // last entry is { 0, 0, 0 }
  transformTuple_t *wp = NULL;

  for (indx = 0; indx < numentries; indx++) {
    if (req[0] == AllSizes4_CPU[indx][0] && req[1] == AllSizes4_CPU[indx][1] &&
        req[2] == AllSizes4_CPU[indx][2] && req[3] == AllSizes4_CPU[indx][3]) {
      // found a match
      wp = (transformTuple_t *)malloc(sizeof(transformTuple_t));
      if (wp != NULL) {
        *wp = fftx_iprdftbat_CPU_Tuples[indx];
      }
      break;
    }
  }

  return wp;
}

//  Run an fftx_iprdftbat_ transform once: run the init functions, run the
//  transform and finally tear down by calling the destroy function.
//  Accepts fftx::point_t<4> specifying size, and pointers to the output
//  (returned) data and the input data.

void fftx_iprdftbat_cpu_Run(fftx::point_t<4> req, double *output,
                            double *input) {
  transformTuple_t *wp = fftx_iprdftbat_cpu_Tuple(req);
  if (wp == NULL)
    //  Requested size not found -- just return
    return;

  //  Call the init function
  (*wp->initfp)();
  //  checkCudaErrors ( cudaGetLastError () );

  (*wp->runfp)(output, input);
  //  checkCudaErrors ( cudaGetLastError () );

  //  Tear down / cleanup
  (*wp->destroyfp)();
  //  checkCudaErrors ( cudaGetLastError () );

  return;
}

//  Host-to-Device C/CUDA/HIP wrapper functions to permit Python to call the
//  kernels.

extern "C" {

int fftx_iprdftbat_cpu_python_init_wrapper(int *req) {
  //  Get the tuple for the requested size
  fftx::point_t<4> rsz;
  rsz[0] = req[0];
  rsz[1] = req[1];
  rsz[2] = req[2];
  rsz[3] = req[3];
  transformTuple_t *wp = fftx_iprdftbat_cpu_Tuple(rsz);
  if (wp == NULL)
    //  Requested size not found -- return false
    return 0;

  //  Call the init function
  (*wp->initfp)();
  return 1;
}

void fftx_iprdftbat_cpu_python_run_wrapper(int *req, double *output,
                                           double *input) {
  //  Get the tuple for the requested size
  fftx::point_t<4> rsz;
  rsz[0] = req[0];
  rsz[1] = req[1];
  rsz[2] = req[2];
  rsz[3] = req[3];
  transformTuple_t *wp = fftx_iprdftbat_cpu_Tuple(rsz);
  if (wp == NULL)
    //  Requested size not found -- just return
    return;

  //  Call the run function
  (*wp->runfp)(output, input);
  return;
}

void fftx_iprdftbat_cpu_python_destroy_wrapper(int *req) {
  //  Get the tuple for the requested size
  fftx::point_t<4> rsz;
  rsz[0] = req[0];
  rsz[1] = req[1];
  rsz[2] = req[2];
  rsz[3] = req[3];
  transformTuple_t *wp = fftx_iprdftbat_cpu_Tuple(rsz);
  if (wp == NULL)
    //  Requested size not found -- just return
    return;

  //  Tear down / cleanup
  (*wp->destroyfp)();
  return;
}
}
