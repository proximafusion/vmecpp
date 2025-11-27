# Compute backend abstraction layer
# CPU backend is always built; CUDA backend is conditionally built

list(APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend.h
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend_cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend_cpu.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend_factory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend_factory.cc
)

# CUDA backend header is always included for compile compatibility
# (the implementation is built separately in the root CMakeLists.txt)
if(VMECPP_ENABLE_CUDA)
  list(APPEND vmecpp_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/compute_backend_cuda.h
  )
endif()

set(vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
