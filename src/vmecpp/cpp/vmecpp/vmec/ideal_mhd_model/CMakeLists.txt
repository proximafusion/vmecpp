list (APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_toroidal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_toroidal.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ideal_mhd_model.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ideal_mhd_model.h
)
# The GPU iteration body; one set of units for both backends, tagged
# LANGUAGE HIP by the parent CMakeLists under VMECPP_USE_HIP.
if(VMECPP_USE_CUDA OR VMECPP_USE_HIP)
  list(APPEND vmecpp_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal_cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal_cuda_common.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal_cuda_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal_cuda_state.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal_cuda_transform.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal_cuda_physics.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal_cuda_control.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal_cuda_io.cu
  )
  if(VMECPP_USE_HIP)
    list(APPEND vmecpp_sources ${CMAKE_CURRENT_SOURCE_DIR}/hip_compat.cuh)
  endif()
endif()
set (vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
