//
//  Copyright (c) 2018-2025, Carnegie Mellon University
//  All rights reserved.
//
//  See LICENSE file for information.
//

#include <stdlib.h>
#include <string.h>

static char fftx_prdftbat_MetaData[] =
    "!!START_METADATA!!\
{\
  \"TransformTypes\": [ \"PRDFTBAT\" ],\
  \"Transforms\": [ \
    {  \"Dimensions\": [ 16 ],\
       \"BatchSize\": 72,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_16_bat_72_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_16_bat_72_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_16_bat_72_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 16 ],\
       \"BatchSize\": 96,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_16_bat_96_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_16_bat_96_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_16_bat_96_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 16 ],\
       \"BatchSize\": 120,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_16_bat_120_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_16_bat_120_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_16_bat_120_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 16 ],\
       \"BatchSize\": 144,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_16_bat_144_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_16_bat_144_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_16_bat_144_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 16 ],\
       \"BatchSize\": 168,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_16_bat_168_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_16_bat_168_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_16_bat_168_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 16 ],\
       \"BatchSize\": 192,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_16_bat_192_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_16_bat_192_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_16_bat_192_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 16 ],\
       \"BatchSize\": 216,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_16_bat_216_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_16_bat_216_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_16_bat_216_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 20 ],\
       \"BatchSize\": 72,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_20_bat_72_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_20_bat_72_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_20_bat_72_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 20 ],\
       \"BatchSize\": 96,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_20_bat_96_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_20_bat_96_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_20_bat_96_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 20 ],\
       \"BatchSize\": 120,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_20_bat_120_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_20_bat_120_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_20_bat_120_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 20 ],\
       \"BatchSize\": 144,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_20_bat_144_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_20_bat_144_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_20_bat_144_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 20 ],\
       \"BatchSize\": 168,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_20_bat_168_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_20_bat_168_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_20_bat_168_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 20 ],\
       \"BatchSize\": 192,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_20_bat_192_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_20_bat_192_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_20_bat_192_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 20 ],\
       \"BatchSize\": 216,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_20_bat_216_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_20_bat_216_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_20_bat_216_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 24 ],\
       \"BatchSize\": 72,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_24_bat_72_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_24_bat_72_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_24_bat_72_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 24 ],\
       \"BatchSize\": 96,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_24_bat_96_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_24_bat_96_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_24_bat_96_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 24 ],\
       \"BatchSize\": 120,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_24_bat_120_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_24_bat_120_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_24_bat_120_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 24 ],\
       \"BatchSize\": 144,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_24_bat_144_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_24_bat_144_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_24_bat_144_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 24 ],\
       \"BatchSize\": 168,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_24_bat_168_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_24_bat_168_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_24_bat_168_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 24 ],\
       \"BatchSize\": 192,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_24_bat_192_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_24_bat_192_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_24_bat_192_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 24 ],\
       \"BatchSize\": 216,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_24_bat_216_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_24_bat_216_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_24_bat_216_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 28 ],\
       \"BatchSize\": 72,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_28_bat_72_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_28_bat_72_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_28_bat_72_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 28 ],\
       \"BatchSize\": 96,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_28_bat_96_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_28_bat_96_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_28_bat_96_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 28 ],\
       \"BatchSize\": 120,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_28_bat_120_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_28_bat_120_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_28_bat_120_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 28 ],\
       \"BatchSize\": 144,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_28_bat_144_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_28_bat_144_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_28_bat_144_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 28 ],\
       \"BatchSize\": 168,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_28_bat_168_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_28_bat_168_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_28_bat_168_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 28 ],\
       \"BatchSize\": 192,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_28_bat_192_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_28_bat_192_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_28_bat_192_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 28 ],\
       \"BatchSize\": 216,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_28_bat_216_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_28_bat_216_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_28_bat_216_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 32 ],\
       \"BatchSize\": 72,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_32_bat_72_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_32_bat_72_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_32_bat_72_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 32 ],\
       \"BatchSize\": 96,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_32_bat_96_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_32_bat_96_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_32_bat_96_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 32 ],\
       \"BatchSize\": 120,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_32_bat_120_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_32_bat_120_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_32_bat_120_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 32 ],\
       \"BatchSize\": 144,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_32_bat_144_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_32_bat_144_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_32_bat_144_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 32 ],\
       \"BatchSize\": 168,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_32_bat_168_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_32_bat_168_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_32_bat_168_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 32 ],\
       \"BatchSize\": 192,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_32_bat_192_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_32_bat_192_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_32_bat_192_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 32 ],\
       \"BatchSize\": 216,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_32_bat_216_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_32_bat_216_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_32_bat_216_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 36 ],\
       \"BatchSize\": 72,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_36_bat_72_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_36_bat_72_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_36_bat_72_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 36 ],\
       \"BatchSize\": 96,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_36_bat_96_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_36_bat_96_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_36_bat_96_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 36 ],\
       \"BatchSize\": 120,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_36_bat_120_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_36_bat_120_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_36_bat_120_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 36 ],\
       \"BatchSize\": 144,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_36_bat_144_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_36_bat_144_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_36_bat_144_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 36 ],\
       \"BatchSize\": 168,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_36_bat_168_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_36_bat_168_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_36_bat_168_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 36 ],\
       \"BatchSize\": 192,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_36_bat_192_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_36_bat_192_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_36_bat_192_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 36 ],\
       \"BatchSize\": 216,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_36_bat_216_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_36_bat_216_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_36_bat_216_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 40 ],\
       \"BatchSize\": 72,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_40_bat_72_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_40_bat_72_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_40_bat_72_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 40 ],\
       \"BatchSize\": 96,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_40_bat_96_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_40_bat_96_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_40_bat_96_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 40 ],\
       \"BatchSize\": 120,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_40_bat_120_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_40_bat_120_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_40_bat_120_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 40 ],\
       \"BatchSize\": 144,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_40_bat_144_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_40_bat_144_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_40_bat_144_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 40 ],\
       \"BatchSize\": 168,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_40_bat_168_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_40_bat_168_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_40_bat_168_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 40 ],\
       \"BatchSize\": 192,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_40_bat_192_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_40_bat_192_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_40_bat_192_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    },\
    {  \"Dimensions\": [ 40 ],\
       \"BatchSize\": 216,\
       \"Direction\": \"Forward\",\
       \"Names\": {\
         \"Destroy\": \"destroy_fftx_prdftbat_40_bat_216_APar_APar_CPU\",\
         \"Exec\": \"fftx_prdftbat_40_bat_216_APar_APar_CPU\",\
         \"Init\": \"init_fftx_prdftbat_40_bat_216_APar_APar_CPU\" },\
       \"Platform\": \"CPU\",\
       \"Precision\": \"Double\",\
       \"ReadStride\": \"APar\",\
       \"WriteStride\": \"APar\",\
       \"TransformType\": \"PRDFTBAT\"\
    }    ]\
}\
!!END_METADATA!!";

//  The metadata table is compiled into the library (and thus readable by
//  scanning file, without having to load the library). Add a simple function to
//  get the metadata (for debug purposes).

char *fftx_prdftbat_cpu_GetMetaData() {
  char *wp = (char *)malloc(strlen(fftx_prdftbat_MetaData) + 1);
  if (wp != NULL) strcpy(wp, fftx_prdftbat_MetaData);

  return wp;
}
