#ifndef fftx_prdftbat_LIB_cpu_HEADER_INCLUDED
#define fftx_prdftbat_LIB_cpu_HEADER_INCLUDED

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

extern "C" {
extern void init_fftx_prdftbat_16_bat_72_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_16_bat_72_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_16_bat_72_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_16_bat_96_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_16_bat_96_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_16_bat_96_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_16_bat_120_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_16_bat_120_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_16_bat_120_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_16_bat_144_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_16_bat_144_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_16_bat_144_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_16_bat_168_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_16_bat_168_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_16_bat_168_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_16_bat_192_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_16_bat_192_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_16_bat_192_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_16_bat_216_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_16_bat_216_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_16_bat_216_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_20_bat_72_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_20_bat_72_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_20_bat_72_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_20_bat_96_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_20_bat_96_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_20_bat_96_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_20_bat_120_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_20_bat_120_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_20_bat_120_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_20_bat_144_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_20_bat_144_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_20_bat_144_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_20_bat_168_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_20_bat_168_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_20_bat_168_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_20_bat_192_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_20_bat_192_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_20_bat_192_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_20_bat_216_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_20_bat_216_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_20_bat_216_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_24_bat_72_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_24_bat_72_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_24_bat_72_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_24_bat_96_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_24_bat_96_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_24_bat_96_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_24_bat_120_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_24_bat_120_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_24_bat_120_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_24_bat_144_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_24_bat_144_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_24_bat_144_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_24_bat_168_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_24_bat_168_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_24_bat_168_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_24_bat_192_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_24_bat_192_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_24_bat_192_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_24_bat_216_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_24_bat_216_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_24_bat_216_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_28_bat_72_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_28_bat_72_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_28_bat_72_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_28_bat_96_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_28_bat_96_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_28_bat_96_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_28_bat_120_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_28_bat_120_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_28_bat_120_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_28_bat_144_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_28_bat_144_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_28_bat_144_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_28_bat_168_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_28_bat_168_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_28_bat_168_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_28_bat_192_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_28_bat_192_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_28_bat_192_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_28_bat_216_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_28_bat_216_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_28_bat_216_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_32_bat_72_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_32_bat_72_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_32_bat_72_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_32_bat_96_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_32_bat_96_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_32_bat_96_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_32_bat_120_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_32_bat_120_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_32_bat_120_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_32_bat_144_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_32_bat_144_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_32_bat_144_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_32_bat_168_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_32_bat_168_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_32_bat_168_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_32_bat_192_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_32_bat_192_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_32_bat_192_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_32_bat_216_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_32_bat_216_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_32_bat_216_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_36_bat_72_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_36_bat_72_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_36_bat_72_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_36_bat_96_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_36_bat_96_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_36_bat_96_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_36_bat_120_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_36_bat_120_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_36_bat_120_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_36_bat_144_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_36_bat_144_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_36_bat_144_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_36_bat_168_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_36_bat_168_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_36_bat_168_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_36_bat_192_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_36_bat_192_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_36_bat_192_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_36_bat_216_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_36_bat_216_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_36_bat_216_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_40_bat_72_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_40_bat_72_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_40_bat_72_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_40_bat_96_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_40_bat_96_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_40_bat_96_APar_APar_CPU(double *output,
                                                  double *input);
}

extern "C" {
extern void init_fftx_prdftbat_40_bat_120_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_40_bat_120_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_40_bat_120_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_40_bat_144_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_40_bat_144_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_40_bat_144_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_40_bat_168_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_40_bat_168_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_40_bat_168_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_40_bat_192_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_40_bat_192_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_40_bat_192_APar_APar_CPU(double *output,
                                                   double *input);
}

extern "C" {
extern void init_fftx_prdftbat_40_bat_216_APar_APar_CPU();
}
extern "C" {
extern void destroy_fftx_prdftbat_40_bat_216_APar_APar_CPU();
}
extern "C" {
extern void fftx_prdftbat_40_bat_216_APar_APar_CPU(double *output,
                                                   double *input);
}

static transformTuple_t fftx_prdftbat_CPU_Tuples[] = {
    {init_fftx_prdftbat_16_bat_72_APar_APar_CPU,
     destroy_fftx_prdftbat_16_bat_72_APar_APar_CPU,
     fftx_prdftbat_16_bat_72_APar_APar_CPU},
    {init_fftx_prdftbat_16_bat_96_APar_APar_CPU,
     destroy_fftx_prdftbat_16_bat_96_APar_APar_CPU,
     fftx_prdftbat_16_bat_96_APar_APar_CPU},
    {init_fftx_prdftbat_16_bat_120_APar_APar_CPU,
     destroy_fftx_prdftbat_16_bat_120_APar_APar_CPU,
     fftx_prdftbat_16_bat_120_APar_APar_CPU},
    {init_fftx_prdftbat_16_bat_144_APar_APar_CPU,
     destroy_fftx_prdftbat_16_bat_144_APar_APar_CPU,
     fftx_prdftbat_16_bat_144_APar_APar_CPU},
    {init_fftx_prdftbat_16_bat_168_APar_APar_CPU,
     destroy_fftx_prdftbat_16_bat_168_APar_APar_CPU,
     fftx_prdftbat_16_bat_168_APar_APar_CPU},
    {init_fftx_prdftbat_16_bat_192_APar_APar_CPU,
     destroy_fftx_prdftbat_16_bat_192_APar_APar_CPU,
     fftx_prdftbat_16_bat_192_APar_APar_CPU},
    {init_fftx_prdftbat_16_bat_216_APar_APar_CPU,
     destroy_fftx_prdftbat_16_bat_216_APar_APar_CPU,
     fftx_prdftbat_16_bat_216_APar_APar_CPU},
    {init_fftx_prdftbat_20_bat_72_APar_APar_CPU,
     destroy_fftx_prdftbat_20_bat_72_APar_APar_CPU,
     fftx_prdftbat_20_bat_72_APar_APar_CPU},
    {init_fftx_prdftbat_20_bat_96_APar_APar_CPU,
     destroy_fftx_prdftbat_20_bat_96_APar_APar_CPU,
     fftx_prdftbat_20_bat_96_APar_APar_CPU},
    {init_fftx_prdftbat_20_bat_120_APar_APar_CPU,
     destroy_fftx_prdftbat_20_bat_120_APar_APar_CPU,
     fftx_prdftbat_20_bat_120_APar_APar_CPU},
    {init_fftx_prdftbat_20_bat_144_APar_APar_CPU,
     destroy_fftx_prdftbat_20_bat_144_APar_APar_CPU,
     fftx_prdftbat_20_bat_144_APar_APar_CPU},
    {init_fftx_prdftbat_20_bat_168_APar_APar_CPU,
     destroy_fftx_prdftbat_20_bat_168_APar_APar_CPU,
     fftx_prdftbat_20_bat_168_APar_APar_CPU},
    {init_fftx_prdftbat_20_bat_192_APar_APar_CPU,
     destroy_fftx_prdftbat_20_bat_192_APar_APar_CPU,
     fftx_prdftbat_20_bat_192_APar_APar_CPU},
    {init_fftx_prdftbat_20_bat_216_APar_APar_CPU,
     destroy_fftx_prdftbat_20_bat_216_APar_APar_CPU,
     fftx_prdftbat_20_bat_216_APar_APar_CPU},
    {init_fftx_prdftbat_24_bat_72_APar_APar_CPU,
     destroy_fftx_prdftbat_24_bat_72_APar_APar_CPU,
     fftx_prdftbat_24_bat_72_APar_APar_CPU},
    {init_fftx_prdftbat_24_bat_96_APar_APar_CPU,
     destroy_fftx_prdftbat_24_bat_96_APar_APar_CPU,
     fftx_prdftbat_24_bat_96_APar_APar_CPU},
    {init_fftx_prdftbat_24_bat_120_APar_APar_CPU,
     destroy_fftx_prdftbat_24_bat_120_APar_APar_CPU,
     fftx_prdftbat_24_bat_120_APar_APar_CPU},
    {init_fftx_prdftbat_24_bat_144_APar_APar_CPU,
     destroy_fftx_prdftbat_24_bat_144_APar_APar_CPU,
     fftx_prdftbat_24_bat_144_APar_APar_CPU},
    {init_fftx_prdftbat_24_bat_168_APar_APar_CPU,
     destroy_fftx_prdftbat_24_bat_168_APar_APar_CPU,
     fftx_prdftbat_24_bat_168_APar_APar_CPU},
    {init_fftx_prdftbat_24_bat_192_APar_APar_CPU,
     destroy_fftx_prdftbat_24_bat_192_APar_APar_CPU,
     fftx_prdftbat_24_bat_192_APar_APar_CPU},
    {init_fftx_prdftbat_24_bat_216_APar_APar_CPU,
     destroy_fftx_prdftbat_24_bat_216_APar_APar_CPU,
     fftx_prdftbat_24_bat_216_APar_APar_CPU},
    {init_fftx_prdftbat_28_bat_72_APar_APar_CPU,
     destroy_fftx_prdftbat_28_bat_72_APar_APar_CPU,
     fftx_prdftbat_28_bat_72_APar_APar_CPU},
    {init_fftx_prdftbat_28_bat_96_APar_APar_CPU,
     destroy_fftx_prdftbat_28_bat_96_APar_APar_CPU,
     fftx_prdftbat_28_bat_96_APar_APar_CPU},
    {init_fftx_prdftbat_28_bat_120_APar_APar_CPU,
     destroy_fftx_prdftbat_28_bat_120_APar_APar_CPU,
     fftx_prdftbat_28_bat_120_APar_APar_CPU},
    {init_fftx_prdftbat_28_bat_144_APar_APar_CPU,
     destroy_fftx_prdftbat_28_bat_144_APar_APar_CPU,
     fftx_prdftbat_28_bat_144_APar_APar_CPU},
    {init_fftx_prdftbat_28_bat_168_APar_APar_CPU,
     destroy_fftx_prdftbat_28_bat_168_APar_APar_CPU,
     fftx_prdftbat_28_bat_168_APar_APar_CPU},
    {init_fftx_prdftbat_28_bat_192_APar_APar_CPU,
     destroy_fftx_prdftbat_28_bat_192_APar_APar_CPU,
     fftx_prdftbat_28_bat_192_APar_APar_CPU},
    {init_fftx_prdftbat_28_bat_216_APar_APar_CPU,
     destroy_fftx_prdftbat_28_bat_216_APar_APar_CPU,
     fftx_prdftbat_28_bat_216_APar_APar_CPU},
    {init_fftx_prdftbat_32_bat_72_APar_APar_CPU,
     destroy_fftx_prdftbat_32_bat_72_APar_APar_CPU,
     fftx_prdftbat_32_bat_72_APar_APar_CPU},
    {init_fftx_prdftbat_32_bat_96_APar_APar_CPU,
     destroy_fftx_prdftbat_32_bat_96_APar_APar_CPU,
     fftx_prdftbat_32_bat_96_APar_APar_CPU},
    {init_fftx_prdftbat_32_bat_120_APar_APar_CPU,
     destroy_fftx_prdftbat_32_bat_120_APar_APar_CPU,
     fftx_prdftbat_32_bat_120_APar_APar_CPU},
    {init_fftx_prdftbat_32_bat_144_APar_APar_CPU,
     destroy_fftx_prdftbat_32_bat_144_APar_APar_CPU,
     fftx_prdftbat_32_bat_144_APar_APar_CPU},
    {init_fftx_prdftbat_32_bat_168_APar_APar_CPU,
     destroy_fftx_prdftbat_32_bat_168_APar_APar_CPU,
     fftx_prdftbat_32_bat_168_APar_APar_CPU},
    {init_fftx_prdftbat_32_bat_192_APar_APar_CPU,
     destroy_fftx_prdftbat_32_bat_192_APar_APar_CPU,
     fftx_prdftbat_32_bat_192_APar_APar_CPU},
    {init_fftx_prdftbat_32_bat_216_APar_APar_CPU,
     destroy_fftx_prdftbat_32_bat_216_APar_APar_CPU,
     fftx_prdftbat_32_bat_216_APar_APar_CPU},
    {init_fftx_prdftbat_36_bat_72_APar_APar_CPU,
     destroy_fftx_prdftbat_36_bat_72_APar_APar_CPU,
     fftx_prdftbat_36_bat_72_APar_APar_CPU},
    {init_fftx_prdftbat_36_bat_96_APar_APar_CPU,
     destroy_fftx_prdftbat_36_bat_96_APar_APar_CPU,
     fftx_prdftbat_36_bat_96_APar_APar_CPU},
    {init_fftx_prdftbat_36_bat_120_APar_APar_CPU,
     destroy_fftx_prdftbat_36_bat_120_APar_APar_CPU,
     fftx_prdftbat_36_bat_120_APar_APar_CPU},
    {init_fftx_prdftbat_36_bat_144_APar_APar_CPU,
     destroy_fftx_prdftbat_36_bat_144_APar_APar_CPU,
     fftx_prdftbat_36_bat_144_APar_APar_CPU},
    {init_fftx_prdftbat_36_bat_168_APar_APar_CPU,
     destroy_fftx_prdftbat_36_bat_168_APar_APar_CPU,
     fftx_prdftbat_36_bat_168_APar_APar_CPU},
    {init_fftx_prdftbat_36_bat_192_APar_APar_CPU,
     destroy_fftx_prdftbat_36_bat_192_APar_APar_CPU,
     fftx_prdftbat_36_bat_192_APar_APar_CPU},
    {init_fftx_prdftbat_36_bat_216_APar_APar_CPU,
     destroy_fftx_prdftbat_36_bat_216_APar_APar_CPU,
     fftx_prdftbat_36_bat_216_APar_APar_CPU},
    {init_fftx_prdftbat_40_bat_72_APar_APar_CPU,
     destroy_fftx_prdftbat_40_bat_72_APar_APar_CPU,
     fftx_prdftbat_40_bat_72_APar_APar_CPU},
    {init_fftx_prdftbat_40_bat_96_APar_APar_CPU,
     destroy_fftx_prdftbat_40_bat_96_APar_APar_CPU,
     fftx_prdftbat_40_bat_96_APar_APar_CPU},
    {init_fftx_prdftbat_40_bat_120_APar_APar_CPU,
     destroy_fftx_prdftbat_40_bat_120_APar_APar_CPU,
     fftx_prdftbat_40_bat_120_APar_APar_CPU},
    {init_fftx_prdftbat_40_bat_144_APar_APar_CPU,
     destroy_fftx_prdftbat_40_bat_144_APar_APar_CPU,
     fftx_prdftbat_40_bat_144_APar_APar_CPU},
    {init_fftx_prdftbat_40_bat_168_APar_APar_CPU,
     destroy_fftx_prdftbat_40_bat_168_APar_APar_CPU,
     fftx_prdftbat_40_bat_168_APar_APar_CPU},
    {init_fftx_prdftbat_40_bat_192_APar_APar_CPU,
     destroy_fftx_prdftbat_40_bat_192_APar_APar_CPU,
     fftx_prdftbat_40_bat_192_APar_APar_CPU},
    {init_fftx_prdftbat_40_bat_216_APar_APar_CPU,
     destroy_fftx_prdftbat_40_bat_216_APar_APar_CPU,
     fftx_prdftbat_40_bat_216_APar_APar_CPU},
    {NULL, NULL, NULL}};

//  Entries in AllSizes4 table:  { FFT length, #batches, read stride type, write
//  stride type }

static fftx::point_t<4> AllSizes4_CPU[] = {
    {16, 72, 0, 0},  {16, 96, 0, 0},  {16, 120, 0, 0}, {16, 144, 0, 0},
    {16, 168, 0, 0}, {16, 192, 0, 0}, {16, 216, 0, 0}, {20, 72, 0, 0},
    {20, 96, 0, 0},  {20, 120, 0, 0}, {20, 144, 0, 0}, {20, 168, 0, 0},
    {20, 192, 0, 0}, {20, 216, 0, 0}, {24, 72, 0, 0},  {24, 96, 0, 0},
    {24, 120, 0, 0}, {24, 144, 0, 0}, {24, 168, 0, 0}, {24, 192, 0, 0},
    {24, 216, 0, 0}, {28, 72, 0, 0},  {28, 96, 0, 0},  {28, 120, 0, 0},
    {28, 144, 0, 0}, {28, 168, 0, 0}, {28, 192, 0, 0}, {28, 216, 0, 0},
    {32, 72, 0, 0},  {32, 96, 0, 0},  {32, 120, 0, 0}, {32, 144, 0, 0},
    {32, 168, 0, 0}, {32, 192, 0, 0}, {32, 216, 0, 0}, {36, 72, 0, 0},
    {36, 96, 0, 0},  {36, 120, 0, 0}, {36, 144, 0, 0}, {36, 168, 0, 0},
    {36, 192, 0, 0}, {36, 216, 0, 0}, {40, 72, 0, 0},  {40, 96, 0, 0},
    {40, 120, 0, 0}, {40, 144, 0, 0}, {40, 168, 0, 0}, {40, 192, 0, 0},
    {40, 216, 0, 0}, {0, 0, 0, 0}};

#endif
