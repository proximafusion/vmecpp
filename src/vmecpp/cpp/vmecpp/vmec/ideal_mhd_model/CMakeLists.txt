list (APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_data.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_toroidal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dft_toroidal.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal.h
  ${CMAKE_CURRENT_SOURCE_DIR}/fft_toroidal.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ideal_mhd_model.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ideal_mhd_model.h
)
if(VMECPP_USE_CUDA)
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
endif()
set (vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)

# Opt-in CUDA-Enzyme Jacobian JVP test (VMECPP_ENABLE_CUDA_ENZYME). A standalone
# executable, independent of vmecpp_core: it compiles the device JVP kernel and
# a host test that differentiates the same shared HalfGridJacobianPoint on the
# CPU as an exact oracle, and validates both against finite differences. Enzyme
# runs as an -O2 optimization-time pass, so both the CUDA and the host
# translation units need -O2 and the plugin attached.
if(VMECPP_ENABLE_CUDA_ENZYME)
  add_executable(cuda_enzyme_jacobian_jvp_test
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_enzyme_jacobian_jvp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_enzyme_jacobian_jvp_test.cc)
  target_include_directories(cuda_enzyme_jacobian_jvp_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/vmecpp/cpp)
  target_compile_options(cuda_enzyme_jacobian_jvp_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-fpass-plugin=${VMECPP_ENZYME_PLUGIN}>
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CXX>:-fpass-plugin=${VMECPP_ENZYME_PLUGIN}>)
  set_target_properties(cuda_enzyme_jacobian_jvp_test PROPERTIES
    CUDA_STANDARD 20 CXX_STANDARD 20)
  target_link_libraries(cuda_enzyme_jacobian_jvp_test PRIVATE CUDA::cudart)
  add_test(NAME cuda_enzyme_jacobian_jvp
    COMMAND cuda_enzyme_jacobian_jvp_test)
endif()
