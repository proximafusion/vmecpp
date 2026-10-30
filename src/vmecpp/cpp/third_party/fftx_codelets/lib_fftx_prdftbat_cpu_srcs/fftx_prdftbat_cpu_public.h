#ifndef fftx_prdftbat_PUBLIC_cpu_HEADER_INCLUDED
#define fftx_prdftbat_PUBLIC_cpu_HEADER_INCLUDED

//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for information.
//

#include "fftx_minimal.hpp"

#ifndef FFTX_INITTRANSFORMFUNC
#define FFTX_INITTRANSFORMFUNC
typedef void (*initTransformFunc)(void);
#endif

#ifndef FFTX_DESTROYTRANSFORMFUNC
#define FFTX_DESTROYTRANSFORMFUNC
typedef void (*destroyTransformFunc)(void);
#endif

#ifndef FFTX_RUNTRANSFORMFUNC
#define FFTX_RUNTRANSFORMFUNC
typedef void (*runTransformFunc)(double *output, double *input);
#endif

#ifndef FFTX_TRANSFORMTUPLE_T
#define FFTX_TRANSFORMTUPLE_T
typedef struct transformTuple {
  initTransformFunc initfp;
  destroyTransformFunc destroyfp;
  runTransformFunc runfp;
} transformTuple_t;
#endif

//  Query the list of sizes available from the library; returns a pointer to an
//  array of length N + 1, where N is the number of unique instances of the
//  transform in the library.  Each element is a struct of type
//  fftx::point_t<4> specifying FFT length, # batches, read-stride type and
//  write-stride type

fftx::point_t<4> *fftx_prdftbat_cpu_QuerySizes();
#define fftx_prdftbat_QuerySizes fftx_prdftbat_cpu_QuerySizes

//  Run an fftx_prdftbat_ transform once: run the init functions, run the,
//  transform and finally tear down by calling the destroy function.
//  Accepts fftx::point_t<4> specifying size, and pointers to the output
//  (returned) data and the input data.

void fftx_prdftbat_cpu_Run(fftx::point_t<4> req, double *output, double *input);
#define fftx_prdftbat_Run fftx_prdftbat_cpu_Run

//  Get a transform tuple -- a set of pointers to the init, destroy, and run
//  functions for a specific size fftx_prdftbat_ transform.  Using this
//  information the user may call the init function to setup for the transform,
//  then run the transform repeatedly, and finally tear down (using destroy
//  function).

transformTuple_t *fftx_prdftbat_cpu_Tuple(fftx::point_t<4> req);
#define fftx_prdftbat_Tuple fftx_prdftbat_cpu_Tuple

//  The metadata table is compiled into the library (and thus readable by
//  scanning the file, without having to load the library). Add a simple
//  function to get the metadata (for debug purposes).

char *fftx_prdftbat_cpu_GetMetaData();

//  Wrapper functions to allow python to call CUDA/HIP GPU code.

extern "C" {

int fftx_prdftbat_cpu_python_init_wrapper(int *req);
void fftx_prdftbat_cpu_python_run_wrapper(int *req, double *output,
                                          double *input);
void fftx_prdftbat_cpu_python_destroy_wrapper(int *req);
}

#endif
