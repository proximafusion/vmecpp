# Compute backend abstraction layer
# CPU backend is always built; CUDA backend is conditionally built

list(APPEND vmecpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend.h
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend_cpu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend_cpu.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend_factory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/compute_backend_factory.cc
)

# CUDA backend headers and pure C++ sources
# (CUDA implementations are built separately in the root CMakeLists.txt)
if(VMECPP_ENABLE_CUDA)
  list(APPEND vmecpp_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/compute_backend_cuda.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/tile_memory_budget.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/tile_scheduler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/cuda_memory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/tiled_compute_backend_cuda.h
  )
endif()

set(vmecpp_sources "${vmecpp_sources}" PARENT_SCOPE)
